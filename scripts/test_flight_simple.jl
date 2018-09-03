using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration
using POMDPs
using POMDPModels
using POMDPToolbox
using StaticArrays
using JLD
using Distributions
using HitchhikingDrones

flight_policy_name = ARGS[1]

uav_dynamics = MultiRotorUAVDynamicsModel(MDP_TIMESTEP, ACC_NOISE_STD, HOVER_COEFFICIENT, FLIGHT_COEFFICIENT)
flight_mdp = UnconstrainedFlightMDP(uav_dynamics, 1.0)

curr_vect = SVector{4,Float64}(0.0,0.0,0.0,0.0)
curr_state = convert_s(MultiRotorUAVState,curr_vect,flight_mdp)
goal_pos = [0.47,-0.63]

flight_policy = load(flight_policy_name,"flight_policy")
cost = 0.0

while true

    rel_uavstate = MultiRotorUAVState(curr_state.x - goal_pos[1],curr_state.y - goal_pos[2],
        curr_state.xdot, curr_state.ydot)
    println(value(flight_policy,rel_uavstate))
    best_action = action(flight_policy, rel_uavstate)
    curr_action = best_action.uav_action
    println(curr_action)
    next_s = next_state(uav_dynamics, curr_state, curr_action)

    cost += MDP_TIMESTEP*TIME_COEFFICIENT + dynamics_cost(uav_dynamics, curr_state, next_s)

    curr_state = deepcopy(next_s)

    curr_vect = convert_s(SVector{4,Float64}, curr_state, flight_mdp)

    println(curr_vect)

    if norm(curr_vect[1:2] - goal_pos) < MDP_TIMESTEP*HOP_DISTANCE_THRESHOLD && norm(curr_vect[3:4]) < XYDOT_HOP_THRESH
        println("SUCCESS!")
        break
    end

    readline()
end

println("TOTAL COST - ",cost)
