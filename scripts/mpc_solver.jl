using StaticArrays
using JSON
using HitchhikingDrones
using Logging

@enum MODE GRAPH_PLAN=1 FLIGHT=2 COAST=3
Logging.configure(level=WARNING)

ep_file = ARGS[1]
to_log = ARGS[2]

episode_dict = Dict()
open(ep_file,"r") do f
    global episode_dict
    episode_dict = JSON.parse(f)
end

start_pos = Point(episode_dict["start_pos"][1], episode_dict["start_pos"][2])
goal_pos = Point(episode_dict["goal_pos"][1], episode_dict["goal_pos"][2])

# Get episode length
num_epochs = episode_dict["num_epochs"]-1

# Define Mersenne Twister for reproducibility
rng = MersenneTwister(2)

uav_dynamics = MultiRotorUAVDynamicsModel(MDP_TIMESTEP, ACC_NOISE_STD)
drone = Drone()

# Create SDMC Simulator and get initial epoch
sdmc_sim = SDMCSimulator(episode_dict["epochs"], uav_dynamics, start_pos, goal_pos, rng)

# Create graph solution
graph_planner = GraphSolution(drone)
setup_graph(graph_planner, start_pos, goal_pos, get_epoch0_dict(sdmc_sim))

# Flight edge weight fn - only nominal
flight_edge_wt_fn(u,v) = flight_edge_cost_nominal(u, v, drone)

log_output = false
log_soln_dict = Dict()
if to_log == "log"
    log_output = true
    log_fn = ARGS[3]
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

episode_reward = 0.0
is_success = false

used_epochs = 0

for epoch = 1:num_epochs

    if mode == GRAPH_PLAN

        # Check if there is a need to replan from the next start
        if need_to_replan == true
            plan_from_next_start(graph_planner, flight_edge_wt_fn, is_valid_flight_edge)
            need_to_replan = false
            readline()
        end

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
                @assert curr_state.on_car == true
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
        @assert curr_state.on_car == false

        if curr_fin_time < Inf
            # Constrained flight action
            if curr_fin_time < MDP_TIMESTEP
                @assert next_vertex.is_car == true

                # Go for the hop if it can
                if curr_dist < MDP_TIMESTEP*HOP_DISTANCE_THRESHOLD
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
            end
        else
            # Unconstrained flight action
            sdmc_action = get_flight_mpc_action_multirotor(curr_state.uav_state, next_vertex, HORIZON_LIM)
        end

        # TODO : Alternative is to put replan=true
    else
        @assert curr_state.on_car == true
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

    curr_state, reward, is_terminal, epoch_info_dict = step_SDMC(sdmc_sim, sdmc_action)
    curr_time = epoch_info_dict["time"]
    episode_reward += reward

    if log_output
        log_soln_dict["epochs"][epoch+1] = Dict("car-info"=>epoch_info_dict["car-info"], 
                                "drone-info"=>Dict("pos"=>[curr_state.uav_state.x, curr_state.uav_state.y],
                                                    "on_car"=>curr_state.car_id))
    end

    if is_terminal
        println("SUCCESS with reward - ",episode_reward)
        used_epochs = epoch
        is_success = true
        break
    end

    update_cars_with_epoch(graph_planner, epoch_info_dict)

    # If there is a need to replan, check if it is close enough in space (and time) to macro action end/start vertex
    if need_to_replan == true
        println(curr_state)
        # readline()
        mode = GRAPH_PLAN
        # If not on a car, then hopped off or aborted in mid-flight or missed connection right at the end
        # In all such cases add a new drone vertex and replan
        if curr_state.on_car == false
            add_new_start(graph_planner, Point(curr_state.uav_state.x, curr_state.uav_state.y), curr_time)
        else

            # TODO : This only happens when a hop is successful? OR unsuccessful hopoff
            info("[OUTER LOOP]: Successful hop OR (not yet) unsuccessful hopoff")
            # Check that last action was a hopon
            @assert sdmc_action[1] == HOPON

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
    println("Did not succeed!")
    log_soln_dict["success"] = false
else
    log_soln_dict["success"] = true
    if log_output
        open(log_fn,"w") do f
            JSON.print(f, log_soln_dict, 2)
        end
    end
end



























