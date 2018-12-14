using StaticArrays
using JSON
using HitchhikingDrones
using Logging
using Random

## Define Mersenne Twister for reproducibility
rng = MersenneTwister(15)

# NOTE - If the macro action is at a stage outside the bounds of the drone, just fly on the st.line path
# towards the goal till it is within range
@enum MODE GRAPH_PLAN=1 FLIGHT=2 COAST=3

Logging.configure(level=ERROR)

# Example usage /dir/to/data/set1-100-to-1000-(1 to 1000) set-1-100-to-1000-paramset3-mpc 1000 <alpha>
ep_file_prefix = ARGS[1]
out_file_prefix = ARGS[2]
num_files = parse(Int, ARGS[3])
energy_time_alpha = parse(Float64, ARGS[4])
to_log = "nolog"

result_stats_dict = Dict()

for iter = 1:num_files

    episode_dict = Dict()
    ep_file = string(ep_file_prefix,"-",iter,".json")

    println(ep_file)

    open(ep_file,"r") do f
        episode_dict = JSON.parse(f)
    end

    start_pos = Point(episode_dict["start_pos"][1], episode_dict["start_pos"][2])
    goal_pos = Point(episode_dict["goal_pos"][1], episode_dict["goal_pos"][2])

    # Get episode length
    num_epochs = episode_dict["num_epochs"]-1

    # Create dynamics model and drone
    uav_dynamics = MultiRotorUAVDynamicsModel(MDP_TIMESTEP, ACC_NOISE_STD, HOVER_COEFFICIENT, FLIGHT_COEFFICIENT)
    drone = Drone()

    # Create SDMC Simulator and get initial epoch
    sdmc_sim = SDMCSimulator(episode_dict["epochs"], uav_dynamics, start_pos, goal_pos, rng)

    # Create graph solution
    graph_planner = GraphSolution(drone)
    setup_graph(graph_planner, start_pos, goal_pos, get_epoch0_dict(sdmc_sim))

    # Flight edge weight fn - only nominal
    flight_edge_wt_fn(u,v) = flight_edge_cost_nominal(u, v, drone, energy_time_alpha)

    log_output = false
    log_soln_dict = Dict()
    if to_log == "log"
        log_output = true
        log_fn = string(out_file_prefix,"-",iter,"-output.json")
        log_soln_dict["start_pos"] = episode_dict["start_pos"]
        log_soln_dict["goal_pos"] = episode_dict["goal_pos"]
        log_soln_dict["epochs"] = Dict()
        log_soln_dict["epochs"][1] = Dict("car-info"=>episode_dict["epochs"]["0"]["car-info"])
        log_soln_dict["epochs"][1]["drone-info"] = Dict("pos"=>[start_pos.x,start_pos.y], "on_car"=>"")
    end

    # Initialize plan
    mode = GRAPH_PLAN
    need_to_replan = true
    curr_state = sdmc_sim.state
    curr_time = 0.0

    # Metrics to track performance
    episode_reward = 0.0
    dist_flown = 0.0
    is_success = false
    used_epochs = 0
    attempted_hops = 0
    successful_hops = 0
    st_line_dist = point_dist(start_pos, goal_pos)

    for epoch = 1:num_epochs

        if mode == GRAPH_PLAN

            # Check if there is a need to replan from the next start
            if need_to_replan == true
                plan_from_next_start(graph_planner, flight_edge_wt_fn, is_valid_flight_edge)
                need_to_replan = false
                # readline()
            end

            # for mav in graph_planner.future_macro_actions_values
            #     println(mav)
            # end

            # If there is no next macro action, just continue (FOR NOW)
            if graph_planner.has_next_macro_action == false
                warn("No macro action available, just waiting")
                sdmc_action = MultiRotorUAVAction(0.0,0.0)
                need_to_replan = true
            else
                # Decide what the next macro action is and set modes accordingly
                next_macro_edge = graph_planner.future_macro_actions_values[1][1]

                # If it is currently on a car it should be coast, regardless of destination
                if next_macro_edge[1].is_car == true
                    # @assert curr_state.on_car == true
                    if curr_state.on_car == false
                        is_success == false
                        break
                    end
                    mode = COAST
                else
                    mode = FLIGHT
                end
            end
        end

        next_vertex = graph_planner.future_macro_actions_values[1][1][2]
        curr_fin_time = next_vertex.time_stamp - curr_time
        curr_dist = point_dist(Point(curr_state.uav_state.x,curr_state.uav_state.y), next_vertex.pos)

        if mode == FLIGHT
            # @assert curr_state.on_car == false
            if curr_state.on_car == true
                is_success == false
                break
            end

            if curr_fin_time < Inf
                # Constrained flight action
                if curr_fin_time < MDP_TIMESTEP
                    # @assert next_vertex.is_car == true
                    if next_vertex.is_car == false
                        is_success == false
                        break
                    end

                    curr_speed = sqrt(curr_state.uav_state.xdot^2 + curr_state.uav_state.ydot^2)

                    # Go for the hop if it can
                    if curr_dist < MDP_TIMESTEP*HOP_DISTANCE_THRESHOLD && curr_speed < XYDOT_HOP_THRESH
                        sdmc_action = (HOPON, next_vertex.car_id)
                        need_to_replan = true
                    else
                        # Missed connection and didn't abort before - just propagate  dynamics and replan
                        sdmc_action = MultiRotorUAVAction(0.0,0.0)
                        need_to_replan = true
                    end
                else
                    # In all other cases, use MPC
                    curr_fin_horizon = convert(Int64, round(curr_fin_time/MDP_TIMESTEP))

                    # TODO : Checking criteria for abort (OR just replan each time?)
                    sdmc_action = get_flight_mpc_action_multirotor(curr_state.uav_state, next_vertex, curr_fin_horizon)
                    # sdmc_action = get_flight_mpc_action_multirotor(curr_state.uav_state, next_vertex, 10)
                end
            else
                # Unconstrained flight action
                sdmc_action = get_flight_mpc_action_multirotor(curr_state.uav_state, next_vertex, HORIZON_LIM)
                # sdmc_action = get_flight_mpc_action_multirotor(curr_state.uav_state, next_vertex, 10)
            end

            # TODO : Alternative is to put replan=true
        else
            # @assert curr_state.on_car == true
            if curr_state.on_car == false
                is_success == false
                break
            end
            # COAST mode - simpler :P
            # If not in the last stretch, just STAY
            # If by some bug the next vertex is flight, just hopoff
            if next_vertex.is_car == false
                sdmc_action = (HOPOFF,curr_state.car_id)
                need_to_replan = true
            else
                if curr_fin_time < MDP_TIMESTEP
                    sdmc_action = (HOPOFF, curr_state.car_id)
                    need_to_replan = true
                else
                    sdmc_action = (STAY,curr_state.car_id)
                end
            end
        end

        # println(curr_time)
        # println(curr_state)
        # println(next_vertex)
        # println(sdmc_action)
        # readline()

        prev_pos = Point(curr_state.uav_state.x, curr_state.uav_state.y)
        prev_on_car = curr_state.on_car

        curr_state, reward, is_terminal, epoch_info_dict = step_SDMC(sdmc_sim, sdmc_action)

        curr_pos = Point(curr_state.uav_state.x, curr_state.uav_state.y)
        
        # Only count IF ACTUALLY FLOWN!
        if curr_state.on_car == false && prev_on_car == false
            dist_flown += point_dist(prev_pos, curr_pos)
        end

        episode_reward += reward

        curr_time = epoch_info_dict["time"]

        if log_output
            log_soln_dict["epochs"][epoch+1] = Dict("car-info"=>epoch_info_dict["car-info"], 
                                    "drone-info"=>Dict("pos"=>[curr_state.uav_state.x, curr_state.uav_state.y],
                                                        "on_car"=>curr_state.car_id))
        end

        if is_terminal
            used_epochs = epoch
            is_success = true
            break
        end

        update_cars_with_epoch(graph_planner, epoch_info_dict)

        # If there is a need to replan, check if it is close enough in space (and time) to macro action end/start vertex
        if need_to_replan == true
            # readline()

            if typeof(sdmc_action) <: Tuple && sdmc_action[1] == HOPON
                attempted_hops += 1
            end

            mode = GRAPH_PLAN
            # If not on a car, then hopped off or aborted in mid-flight or missed connection right at the end
            # In all such cases add a new drone vertex and replan
            if curr_state.on_car == false
                add_new_start(graph_planner, Point(curr_state.uav_state.x, curr_state.uav_state.y), curr_time)
            else

                # TODO : This only happens when a hop is successful? OR unsuccessful hopoff
                info("[OUTER LOOP]: Successful hop OR (not yet) unsuccessful hopoff")
                # Check that last action was a hopon
                # @assert sdmc_action[1] == HOPON

                successful_hops += 1

                # Check up the next vertex that the car has to pass. IF it is NOT the same as next_vertex
                # then reset it to that and move the time forward 
                if graph_planner.car_map[curr_state.car_id].route_idx_range[1] != next_vertex.idx
                    graph_planner.car_map[curr_state.car_id].route_idx_range[1] = next_vertex.idx
                    graph_planner.next_start_idx = next_vertex.idx
                    next_vertex.last_time_stamp = next_vertex.time_stamp
                    next_vertex.time_stamp = curr_time
                end
            end
        end
    end

    log_soln_dict["num_epochs"] = used_epochs + 1

    if is_success != true
        log_soln_dict["num_epochs"] = num_epochs
        println("Episode ",iter," : Did not succeed!")
        log_soln_dict["success"] = false
    else
        log_soln_dict["num_epochs"] = used_epochs + 1
        println("Episode ",iter," : Succeeded with reward ",episode_reward)
        log_soln_dict["success"] = true
    end

    if log_output
        open(log_fn,"w") do f
            JSON.print(f, log_soln_dict, 2)
        end
    end

    # Update results stats
    result_stats_dict[iter] = Dict("success"=>is_success, "reward"=>episode_reward, 
                                   "distance"=>dist_flown, "sl_dist"=>st_line_dist, "time"=>used_epochs*MDP_TIMESTEP,
                                   "attempted_hops"=>attempted_hops,"successful_hops"=>successful_hops)
end

# Write stats json to file
stats_filename = string(out_file_prefix,"-mpc-solver-stats.json")
open(stats_filename,"w") do f
    JSON.print(f,result_stats_dict,2)
end



























