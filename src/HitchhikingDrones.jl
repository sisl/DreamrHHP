module HitchhikingDrones

# Stdlib Requirements
using Logging
using Printf
using Random
using Statistics

# Common non-stdlib requirements
using DataStructures
using StaticArrays
using Distributions

# For plotting
using Cairo 
using Gadfly
using Colors
using Measures

# Graph planning/MPC stuff
using Graphs
using JuMP
using Ipopt

# POMDP Stuff
using POMDPs
using POMDPModelTools
using POMDPModels
using POMDPPolicies
using POMDPSimulators
using LocalApproximationValueIteration

# Parsing and support
using TOML


# package code goes here

# Dynamics stuff
# export
#     UAVDynamicsModel,
#     UAVState,
#     UAVAction,
#     MultiRotorUAVState,
#     MultiRotorUAVAction,
#     MultiRotorUAVDynamicsModel,
#     sigma_point_states_weights,
#     next_state,
#     dynamics_cost,
#     HOP_ACTION,
#     HOPON,
#     STAY,
#     HOPOFF


# # Graph Planner components
# export
#     remove_last_vertex,
#     astar_shortest_path,
#     astar_light_shortest_path_implicit,
#     SimpleVListGraph,
#     CarDroneVertex,
#     is_valid_flight_edge,
#     coast_edge_cost,
#     flight_edge_cost_nominal,
#     flight_edge_cost_valuefn,
#     GraphSolution,
#     setup_graph,
#     update_cars_with_epoch,
#     add_drone_vertex,
#     add_new_start,
#     update_next_start,
#     revert_new_start,
#     plan_from_next_start,
#     get_future_macro_actions_values,
#     get_flight_mpc_action_multirotor

# # Partial Control MDP stuff
# export
#     ControlledHopOnStateAugmented,
#     HopOffStateAugmented,
#     HopOnAction,
#     HopOffAction,
#     HopOffMDP,
#     ControlledMultiRotorHopOnMDP,
#     PartialControlHopOnOffPolicy,
#     UnconstrainedFlightMDP,
#     hopon_policy_action,
#     hopoff_policy_action,
#     outhor_outdist_action,
#     unconstrained_flight_action,
#     terminalreward,
#     actions,
#     n_actions,
#     discount,
#     reward,
#     generate_sr,
#     transition,
#     isterminal,
#     action_index,
#     generate_start_state,
#     get_state_at_rest,
#     convert_s

# # Parameters for partial control MDP
# export
#     XY_LIM,
#     XYDOT_LIM,
#     ACCELERATION_LIM,
#     ACCELERATION_NUMVALS,
#     HORIZON_LIM,
#     EPSILON,
#     XY_AXISVALS,
#     XYDOT_AXISVALS,
#     XYDOT_HOP_THRESH,
#     MDP_TIMESTEP,
#     ACC_NOISE_STD,
#     CONTROL_TRANSFER_PENALTY,
#     HOP_DISTANCE_THRESHOLD,
#     FLIGHT_COEFFICIENT,
#     TIME_COEFFICIENT,
#     HOVER_COEFFICIENT,
#     NO_HOP_PENALTY,
#     MC_GENERATIVE_NUMSAMPLES,
#     MC_TIME_NUMSAMPLES,
#     MAX_DRONE_SPEED,
#     MAX_CAR_SPEED,
#     MAX_REPLAN_TIMESTEP

# For parameters
export
    ScaleParameters,
    SimTimeParameters,
    CostParameters,
    Parameters,
    parse_scale,
    parse_simtime,
    parse_cost,
    parse_params

# Types information
export
    Point,
    point_dist,
    point_norm,
    equal,
    interpolate,
    Car,
    InactiveCar,
    Drone

# # For simulator
# export
#     HopOnOffSingleCarSimulator,
#     reset_sim,
#     sample_finish_time,
#     step_sim,
#     SDMCState,
#     SDMCAction,
#     SDMCSimulator,
#     step_SDMC,
#     get_epoch0_dict


# General utils
export
    plot_car_route,
    plot_drone_and_active_cars_epoch!,
    log2space_symmetric,
    polyspace_symmetric,
    st_line_reward_time



include("types.jl")
include("parameters.jl")
include("plot_utils.jl")
include("utils.jl")
# include("macro_action_policy/uavdynamics.jl")
# include("macro_action_policy/partial_control_mdp.jl")
# include("graph_plan/astar_visitor_light.jl")
# include("graph_plan/graph_helpers.jl")
# include("graph_plan/graph_solution.jl")
# include("graph_plan/mpc_utils.jl")
# include("simulators/hoponoff_singlecar_simulator.jl")
# include("simulators/sdmc_simulator.jl")

end # module
