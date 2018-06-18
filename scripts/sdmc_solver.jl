using GridInterpolations
using POMDPs
using POMDPModels
using POMDPToolbox
using StaticArrays
using LocalFunctionApproximation
using LocalApproximationValueIteration
using JLD
using JSON
using HitchhikingDrones



# NOTE - If the macro action is at a stage outside the bounds of the drone, just fly on the st.line path
# towards the goal till it is within range

@enum MODE GRAPH_PLAN=1 FLIGHT=2 COAST=3


# Read in the episode filename from args
ep_file = "../data/test-3-120-to-150.json"
hopon_file = "policies/hopon_generative_unitgrid_paramset2.jld"
hopoff_file = "policies/hopoff.jld"

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

# Create dynamics model and MDP
uav_dynamics = MultiRotorUAVDynamicsModel(MDP_TIMESTEP, ACC_NOISE_STD)
pc_hopon_mdp = ControlledMultiRotorHopOnMDP(uav_dynamics)
uc_hopoff_mdp = HopOffMDP()

# Create drone
drone = Drone()

# Create SDMC Simulator and get initial epoch
sdmc_sim = SDMCSimulator(episode_dict["epochs"], uav_dynamics, start_pos, goal_pos, rng)

# Load policies
hopon_policy = load(hopon_file,"policy")
hopoff_policy = load(hopoff_file,"policy")

# Create graph solution

graph_planner = GraphSolution(drone)
setup_graph(graph_planner, start_pos, goal_pos, get_epoch0_dict(sdmc_sim))
flight_edge_wt_fn(u,v) = flight_edge_cost_valuefn(uav_dynamics, hopon_policy, u, v, drone)


# Initialize plan
mode = GRAPH_PLAN
need_to_replan = true
curr_state = sdmc_sim.state
curr_time = 0.0

episode_reward = 0.0
is_success = false


for epoch = 1:num_epochs

    # At the beginning of the epoch, check which mode you are in
    if mode == GRAPH_PLAN

        # Check if there is a need to replan from the next start
        if need_to_replan == true
            @time plan_from_next_start(graph_planner, flight_edge_wt_fn, is_valid_flight_edge)
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
            if curr_fin_time/MDP_TIMESTEP > HORIZON_LIM + 4

                if abs(rel_uavstate.x) > 1.05*XY_LIM || abs(rel_uavstate.y) > 1.05*XY_LIM
                    best_action = outhor_outdist_action(rel_uavstate)
                else
                    aug_outhor_state = ControlledHopOnStateAugmented(rel_uavstate,false,0.)
                    best_action = action(hopon_policy.out_horizon_policy, aug_outhor_state)
                end

                if best_action.control_transfer == true
                    # Lower level action aborts
                    info("Aborting current hopon action")
                    sdmc_action = MultiRotorUAVAction(0.0,0.0)
                    need_to_replan = true
                else
                    # Just get the UAV action
                    sdmc_action = best_action.uavaction
                end
            elseif curr_fin_time < MDP_TIMESTEP
                @assert next_vertex.is_car==true
                # Go for the hop if you can
                if curr_dist < MDP_TIMESTEP*HOP_DISTANCE_THRESHOLD
                    sdmc_action = (HOPON, next_vertex.car_id)
                    need_to_replan = true
                else
                    # Missed connection and didn't abort before - just propagate  dynamics and replan
                    sdmc_action = MultiRotorUAVAction(0.0,0.0)
                    need_to_replan = true
                end
            else
                best_action = hopon_policy_action(hopon_policy, rel_uavstate, curr_fin_time)

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
            sdmc_action = unconstrained_flight_action(rel_uavstate)
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
            if curr_fin_time/MDP_TIMESTEP > 4
                sdmc_action = (STAY,curr_state.car_id)
            else
                best_action = hopoff_policy_action(hopoff_policy, curr_fin_time)
                sdmc_action = (best_action.hopaction, curr_state.car_id)
                if best_action.hopaction == HOPOFF 
                    need_to_replan=true
                end
            end
        end
    end

    # println(curr_time)
    # println(curr_state)
    # println(next_vertex)
    # println(sdmc_action)
    # readline()

    # Invoke action on SDMC simulator and get new current state and time
    curr_state, reward, is_terminal, epoch_info_dict = step_SDMC(sdmc_sim, sdmc_action)
    curr_time = epoch_info_dict["time"]
    episode_reward += reward

    if is_terminal
        println("SUCCESS!")
        is_success = true
        break
    end


    update_cars_with_epoch(graph_planner, epoch_info_dict)

    # If there is a need to replan, check if it is close enough in space (and time) to macro action end/start vertex
    if need_to_replan == true
        println(curr_state)
        readline()
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

if is_success != true
    println("Did not succeed!")
end
