using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration
using POMDPs
using POMDPModels
using POMDPModelTools
using StaticArrays
using JLD2, FileIO
using Random
using LinearAlgebra
using Logging
using Distributions
using PDMats
using Statistics
using HitchhikingDrones

# Uncomment if testing in REPL
# ARGS = String["../data/paramsets/scale-small-test.toml","../data/paramsets/simtime-small-test.toml",
#                 "../data/paramsets/cost-1.toml","../data/policies/test-uf-poly-alpha-0.75.jld2","0.75"]

DISCOUNT = 1.0

scale_file = ARGS[1]
simtime_file = ARGS[2]
cost_file = ARGS[3]
flight_policy_name = ARGS[4]
energy_time_alpha = parse(Float64,ARGS[5])

# First, parse parameter files to get filenames and construct params object
params = parse_params(scale_file=scale_file, simtime_file=simtime_file, cost_file=cost_file)

uav_dynamics = MultiRotorUAVDynamicsModel(params.time_params.MDP_TIMESTEP, params.scale_params.ACC_NOISE_STD, params)
flight_mdp = UnconstrainedFlightMDP{MultiRotorUAVState,MultiRotorUAVAction}(uav_dynamics, DISCOUNT, energy_time_alpha)

curr_vect = SVector{4,Float64}(0.0,0.0,0.0,0.0)
curr_state = convert_s(MultiRotorUAVState,curr_vect,flight_mdp)
goal_pos = [0.47,-0.63]

flight_policy = load_localapproxvi_policy_from_jld2(flight_policy_name)
cost = 0.0

while true
    global curr_state, curr_vect, cost
    rel_uavstate = MultiRotorUAVState(curr_state.x - goal_pos[1],curr_state.y - goal_pos[2],
        curr_state.xdot, curr_state.ydot)
    @show value(flight_policy,rel_uavstate)
    best_action = action(flight_policy, rel_uavstate)
    curr_action = best_action.uav_action
    @show curr_action
    next_s = next_state(uav_dynamics, curr_state, curr_action)

    cost += params.time_params.MDP_TIMESTEP*params.cost_params.TIME_COEFFICIENT + 
            dynamics_cost(uav_dynamics, curr_state, next_s)

    curr_state = deepcopy(next_s)

    curr_vect = convert_s(SVector{4,Float64}, curr_state, flight_mdp)

    @show curr_vect

    if norm(curr_vect[1:2] - goal_pos) < params.time_params.MDP_TIMESTEP*params.scale_params.HOP_DISTANCE_THRESHOLD && 
       norm(curr_vect[3:4]) < params.scale_params.XYDOT_HOP_THRESH
        println("SUCCESS!")
        break
    end

    readline()
end

println("TOTAL COST - ",cost)
