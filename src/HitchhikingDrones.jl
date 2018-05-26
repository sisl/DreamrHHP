__precompile__()
module HitchhikingDrones

using Base
using Graphs
using DataStructures
using StaticArrays
using Distributions
using POMDPModels
using POMDPs
using POMDPToolbox
using LocalApproximationValueIteration


# package code goes here


# Graph Planner components
export
    astar_shortest_path,
    astar_light_shortest_path_implicit,
    SimpleVListGraph,
    CarDroneVertex,
    GraphSolution,
    setup_graph,
    update_cars_with_epoch,
    add_drone_vertex,
    add_new_start,
    update_next_start,
    plan_from_next_start,
    get_future_macro_actions_values

# Dynamics stuff
export
    UAVDynamicsModel,
    UAVState,
    UAVAction,
    MultiRotorUAVState,
    MultiRotorUAVAction,
    MultiRotorUAVDynamicsModel,
    sigma_point_states_weights,
    next_state,
    dynamics_cost


# Partial Control MDP stuff
export
    ControlledHopOnStateAugmented,
    ControlledHopOnState,
    HopOnAction,
    ControlledMultiRotorHopOnMDP,
    PartialControlHopOnPolicy,
    terminalreward,
    actions,
    n_actions,
    discount,
    reward,
    generate_sr,
    isterminal,
    action_index,
    generate_start_state,
    get_state_at_rest

# Parameters for partial control MDP
export
    XY_LIM,
    XYDOT_LIM,
    HORIZON_LIM,
    XY_RES,
    XYDOT_RES,
    MDP_TIMESTEP,
    ACC_NOISE_STD,
    CONTROL_TRANSFER_PENALTY,
    HOP_DISTANCE_THRESHOLD,
    HOP_REWARD

# Types information
export
    Point,
    pointDist,
    point_norm,
    equal,
    interpolate,
    Car,
    Drone

# For simulator
export
    HopOnOffSingleCarSimulator,
    reset_sim,
    sample_finish_time,
    step_sim





include("types.jl")
include("parameters.jl")
include("graph_plan/astar_visitor_light.jl")
include("graph_plan/graph_helpers.jl")
include("graph_plan/graph_solution.jl")
include("macro_action_policy/uavdynamics.jl")
include("macro_action_policy/partial_control_mdp.jl")
include("simulators/hoponoff_singlecar_simulator.jl")

end # module
