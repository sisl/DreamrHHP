__precompile__()
module HitchhikingDrones

using Base
using DataStructures
using Distributions
using POMDPModels
using POMDPs
using POMDPToolbox
using LocalApproximationValueIteration


# package code goes here

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
    generate_start_state

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
    point_norm

# For simulator
export
    HopOnOffSingleCarSimulator,
    reset_sim,
    sample_finish_time,
    step_sim


include("types.jl")
include("parameters.jl")
include("macro_action_policy/uavdynamics.jl")
include("macro_action_policy/partial_control_mdp.jl")
include("simulators/hoponoff_singlecar_simulator.jl")

end # module
