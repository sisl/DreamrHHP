using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration
using POMDPs
using POMDPModels
using POMDPModelTools
using StaticArrays
using JLD2, FileIO
using Random
using Logging
using Distributions
using PDMats
using Statistics
using HitchhikingDrones
using JSON

# julia sdmc_solver.jl hopon_preabort-111-alpha0.375-poly-abort_thresh-0.75-inhor.jld2 hopon_preabort-111-alpha0.375-poly-preabort-outhor.jld2 hopoff-111-inhor.jld2 hopoff-111-outhor.jld2 flight-111-alpha-0.375-poly.jld2 ../data/paramsets/scale-1.toml ../data/paramsets/simtime-1.toml ../data/paramsets/cost-1.toml /scratch/shushman/HitchhikingDrones/set-1-easy/set-1-100-to-1000 1 nolog



## Define Mersenne Twister for reproducibility
rng = MersenneTwister(15)

## Script arguments
## Example usage <policy-args> /dir/to/data/set1-100-to-1000 1000 nolog
# Args for policies to use
hopon_inhor_file = ARGS[1]
hopon_outhor_file = ARGS[2]
hopoff_inhor_file = ARGS[3]
hopoff_outhor_file = ARGS[4]
flight_file = ARGS[5]

# Args for parameter files
scale_file = ARGS[6]
simtime_file = ARGS[7]
cost_file = ARGS[8]

# Other arguments related to evaluation
ep_file_prefix = ARGS[9]
num_files = parse(Int, ARGS[10])
to_log = ARGS[11]

# Load the policies to use
@show "Loading policies"
hopon_policy = load_partialcontrolpolicy(hopon_inhor_file, hopon_outhor_file)
hopoff_policy = load_partialcontrolpolicy(hopoff_inhor_file, hopoff_outhor_file)
flight_policy = load_localapproxvi_policy_from_jld2(flight_file)

# Parse the parameters
params = parse_params(scale_file=scale_file, simtime_file=simtime_file, cost_file=cost_file)

# NOTE - If the macro action is at a stage outside the bounds of the drone, just fly on the st.line path
# towards the goal till it is within range
@enum MODE GRAPH_PLAN=1 FLIGHT=2 COAST=3

result_stats_dict = Dict()

# Iterate over every single episode file
for iter = 1:num_files

    episode_dict = Dict()
    ep_file = string(ep_file_prefix,"-",iter,".json")

    @show ep_file

    open(ep_file,"r") do f
        episode_dict = JSON.parse(f)
    end

    start_pos = Point(episode_dict["start_pos"][1], episode_dict["start_pos"][2])
    goal_pos = Point(episode_dict["goal_pos"][1], episode_dict["goal_pos"][2])

    # Get episode length
    num_epochs = episode_dict["num_epochs"]-1

    # Create dynamics model and drone
    uav_dynamics = MultiRotorUAVDynamicsModel(params.time_params.MDP_TIMESTEP, params.scale_params.ACC_NOISE_STD, params)
    drone = Drone(params)

    # Create SDMC Simulator and get initial epoch
    sdmc_sim = SDMCSimulator(episode_dict["epochs"], uav_dynamics, start_pos, goal_pos, params, rng)

    # Create graph solution
    graph_planner = GraphSolution(drone, params)
    setup_graph(graph_planner, start_pos, goal_pos, get_epoch0_dict(sdmc_sim))

    # Use the value function for flight edge weights
    flight_edge_wt_fn(u,v) = flight_edge_cost_valuefn(uav_dynamics, hopon_policy, flight_policy, u, v, params)
    # flight_edge_wt_fn(u,v) = flight_edge_cost_nominal(u, v, drone)


    log_output = false
    log_soln_dict = Dict()
    if to_log == "log"
        log_output = true
        log_fn = string(ep_file_prefix,"-",iter,"-output.json",)
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

    for epoch = 1:num_epochs

        # At the beginning of the epoch, check which mode you are in
        if mode == GRAPH_PLAN

            # Check if there is a need to replan from the next start
            if need_to_replan == true
                plan_from_next_start(graph_planner, flight_edge_wt_fn, is_valid_flight_edge)
                need_to_replan = false
                # for mav in graph_planner.future_macro_actions_values
                #     println(mav)
                # end
                # readline()
            end

            # If there is no next macro action, just continue (FOR NOW)
            if graph_planner.has_next_macro_action == false
                @warn "No macro action available, just waiting"
                sdmc_action = MultiRotorUAVAction(0.0,0.0)
                need_to_replan = true
            else
                # Decide what the next macro action is and set modes accordingly
                next_macro_edge = graph_planner.future_macro_actions_values[1][1]

                # If it is currently on a car it should be coast, regardless of destination
                if next_macro_edge[1].is_car == true
                    @assert curr_state.on_car == true
                    mode = COAST
                else
                    mode = FLIGHT
                end
            end
        end

        # This is your next vertex
        next_vertex = graph_planner.future_macro_actions_values[1][1][2]
        curr_fin_time = next_vertex.time_stamp - curr_time
        curr_dist = point_dist(Point(curr_state.uav_state.x,curr_state.uav_state.y), next_vertex.pos)

        if mode == FLIGHT
            @assert curr_state.on_car == false
            # set up relative UAV state
            rel_uavstate = MultiRotorUAVState(curr_state.uav_state.x - next_vertex.pos.x,curr_state.uav_state.y - next_vertex.pos.y,
                                              curr_state.uav_state.xdot, curr_state.uav_state.ydot)
            # NOTE - Heuristic for efficiency
            # If the horizon is far out just directly generate the out-horizon action
            if curr_fin_time < Inf
                if curr_fin_time/params.time_params.MDP_TIMESTEP > params.time_params.HORIZON_LIM + 2
                    aug_outhor_state = ControlledHopOnStateAugmented(rel_uavstate,params.time_params.HORIZON_LIM+1)
                    best_action = action(hopon_policy.out_horizon_policy, aug_outhor_state)

                    if best_action.control_transfer == true
                        # Lower level action aborts
                        @info "Aborting current hopon action"
                        sdmc_action = MultiRotorUAVAction(0.0,0.0)
                        need_to_replan = true
                    else
                        # Just get the UAV action
                        sdmc_action = best_action.uavaction
                    end
                elseif curr_fin_time < params.time_params.MDP_TIMESTEP
                    @assert next_vertex.is_car==true
                    curr_speed = get_speed(curr_state.uav_state)
                    # Go for the hop if you can
                    if curr_dist < params.time_params.MDP_TIMESTEP*params.scale_params.HOP_DISTANCE_THRESHOLD && 
                        curr_speed < params.scale_params.XYDOT_HOP_THRESH
                        sdmc_action = (HOPON, next_vertex.car_id)
                        need_to_replan = true
                    else
                        # Missed connection and didn't abort before - just propagate  dynamics and replan
                        sdmc_action = MultiRotorUAVAction(0.0,0.0)
                        need_to_replan = true
                    end
                else
                    best_action = hopon_policy_action(hopon_policy, params, rel_uavstate, curr_fin_time, rng)

                    if best_action.control_transfer == true
                        # Lower level action aborts
                        info("Aborting current hopon action")
                        sdmc_action = MultiRotorUAVAction(0.0,0.0)
                        need_to_replan = true
                    else
                        # Just get the UAV action
                        sdmc_action = best_action.uavaction
                    end
                end
            else
                best_action = action(flight_policy, rel_uavstate)
                sdmc_action = best_action.uav_action
            end

        else
            @assert curr_state.on_car == true
            # COAST mode - simpler :P
            # If not in the last stretch, just STAY
            # If by some bug the next vertex is flight, just hopoff
            if next_vertex.is_car == false
                sdmc_action = (HOPOFF,curr_state.car_id)
                need_to_replan = true
            else 
                best_action = hopoff_policy_action(hopoff_policy, params, curr_fin_time, rng)
                sdmc_action = (best_action.hopaction, curr_state.car_id)
                if best_action.hopaction == HOPOFF 
                    need_to_replan=true
                end
            end
        end

        # println(curr_time)
        # println(curr_state)
        # println(next_vertex)
        # println(sdmc_action)
        # readline()

        prev_pos = Point(curr_state.uav_state.x, curr_state.uav_state.y)

        # Invoke action on SDMC simulator and get new current state and time
        curr_state, reward, is_terminal, epoch_info_dict = step_sim(sdmc_sim, sdmc_action)

        curr_pos = Point(curr_state.uav_state.x, curr_state.uav_state.y)
        dist_flown += point_dist(prev_pos, curr_pos)

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
                @info "[OUTER LOOP]: Successful hop OR (not yet) unsuccessful hopoff"
                # Check that last action was a hopon
                @assert sdmc_action[1] == HOPON

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
        println("Episode ",iter," : Did not succeed!")
        log_soln_dict["success"] = false
    else
        println("Episode ",iter," : Succeeded with reward ",episode_reward)
        log_soln_dict["success"] = true
        if log_output
            open(log_fn,"w") do f
                JSON.print(f, log_soln_dict, 2)
            end
        end
    end

    # Update results stats
    result_stats_dict[iter] = Dict("success"=>is_success, "reward"=>episode_reward, 
                                   "distance"=>dist_flown, "time"=>used_epochs*params.time_params.MDP_TIMESTEP,
                                   "attempted_hops"=>attempted_hops,"successful_hops"=>successful_hops)
end # End for iter 

# Write stats json to file
stats_filename = string(ep_file_prefix,"-testjulia1-sdmc-solver-stats.json")
open(stats_filename,"w") do f
    JSON.print(f,result_stats_dict,2)
end