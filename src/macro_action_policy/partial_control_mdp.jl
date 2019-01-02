"""
The controlled state space of the UAV when it is planning
to track the car and hop on to it. It must try to reach the goal 
position before the car reaches it.
"""
struct ControlledHopOnStateAugmented{US <: UAVState}
    rel_uavstate::US  # RELATIVE to goal
    horizon::Int64    # POMDPs.jl has no explicit interface for finite horizon problems
end

"""
For now, this just has the horizon as the dynamics depend on that of the car.
We want the drone to stay on till the car is close to the transfer point
"""
struct HopOffStateAugmented
    oncar::Bool
    horizon::Int64
end

"""
For constrained flight, a valid action is either a control action for the UAV or abort (transfer to higher layer).
"""
struct HopOnAction{UA <: UAVAction}
    action_idx::Int64
    uavaction::Union{Nothing,UA} 
    control_transfer::Union{Nothing,Bool}
end

"""
For hitchhiking, the only valid action is to stay or hopoff.
"""
struct HopOffAction
    action_idx::Int64
    hopaction::HOP_ACTION
end


################################# HOPOFF MDP ###################################
"""
Defines the hitchhiking MDP
"""
mutable struct HopOffMDP <: POMDPs.MDP{HopOffStateAugmented,HopOffAction}
    actions::Vector{HopOffAction}
    terminal_costs_set::Bool
    params::Parameters
end

"""
Initializes the hitchhiking MDP with the two actions of STAY and HOPOFF
"""
function HopOffMDP(params::Parameters)
    hopoff_actions = [HopOffAction(1,STAY), HopOffAction(2,HOPOFF)]
    return HopOffMDP(hopoff_actions, true, params)
end

# Overridden methods for POMDPs
POMDPs.actions(mdp::HopOffMDP) = mdp.actions
POMDPs.actions(mdp::HopOffMDP, s::HopOffStateAugmented) = mdp.actions
POMDPs.n_actions(mdp::HopOffMDP) = length(mdp.actions)
POMDPs.discount(mdp::HopOffMDP) = 1.0
POMDPs.actionindex(mdp::HopOffMDP, a::HopOffAction) = a.action_idx

"""
Hitchhiking state is terminal if not on car or end of horizon
"""
function POMDPs.isterminal(mdp::HopOffMDP, s::HopOffStateAugmented)
    return (s.oncar == false || s.horizon == 0)
end

"""
Hitchhiking transition function is deterministic
"""
function POMDPs.transition(mdp::HopOffMDP, s::HopOffStateAugmented, a::HopOffAction)

    if a.hopaction == STAY
        return SparseCat([HopOffStateAugmented(true,s.horizon-1)],[1.0])
    else
        return SparseCat([HopOffStateAugmented(false,s.horizon-1)],[1.0])
    end
end

function POMDPs.reward(mdp::HopOffMDP, s::HopOffStateAugmented, a::HopOffAction, sp::HopOffStateAugmented)
    if mdp.terminal_costs_set
        if (sp.horizon == 0 && sp.oncar == true) || (sp.horizon > 0 && sp.oncar == false)
            return -mdp.params.cost_params.NO_HOPOFF_PENALTY
        end
    end
    return -(mdp.params.cost_params.TIME_COEFFICIENT*mdp.params.time_params.MDP_TIMESTEP)
end

"""
Convert hopoff state to 2D float vector
"""
function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, s::HopOffStateAugmented, mdp::HopOffMDP)
  v = SVector{2,Float64}(convert(Float64,s.oncar), convert(Float64,s.horizon))
  return v
end

"""
Convert 2D float vector to state
"""
function POMDPs.convert_s(::Type{HopOffStateAugmented}, v::AbstractVector{Float64}, mdp::HopOffMDP)
  s = HopOffStateAugmented(convert(Bool,v[1]), convert(Int64,v[2]))
  return s
end

################################# MULTI ROTOR MDP #################################
"""
Defines the constrained flight MDP. It is templated on the type of action and dynamics model. The UAV state type is not
explicitly needed by the MDP and is obtained through arguments to methods.
"""
mutable struct ControlledHopOnMDP{US <: UAVState, UA <: UAVAction, UDM <: UAVDynamicsModel} <: POMDPs.MDP{ControlledHopOnStateAugmented{US},HopOnAction{UA}}
    dynamics::UDM
    actions::Vector{HopOnAction{UA}}
    horizon_abort_penalty::Vector{Float64}
    terminal_costs_set::Bool
    no_hop_penalty::Float64
    params::Parameters
    energy_time_alpha::Float64
end

"""
    ControlledHopOnMDP{UA}(dynamics::UDM, energy_time_alpha=0.5) where {UA <: UAVAction, UDM <: UAVDynamicsModel}
    
Initialize the constrained flight MDP with a UAV dynamics model and a tradeoff between energy and time (to be used by the reward function).
It requires the types of state, action, and dynamics
"""
function ControlledHopOnMDP{US,UA}(dynamics::UDM, energy_time_alpha=0.5) where {US <: UAVState, UA <: UAVAction, UDM <: UAVDynamicsModel}

    uav_dynamics_actions = get_uav_dynamics_actions(dynamics)
    hopon_actions = Vector{HopOnAction{UA}}(undef,0)
    idx = 1

    for uavda in uav_dynamics_actions
        push!(hopon_actions, HopOnAction{UA}(idx,uavda,nothing))
        idx+=1
    end
    push!(hopon_actions, HopOnAction{UA}(idx,nothing,true))

    return ControlledHopOnMDP{US, UA, UDM}(dynamics,hopon_actions,Inf*ones(dynamics.params.time_params.HORIZON_LIM),true,Inf,dynamics.params,energy_time_alpha)
end


POMDPs.actions(mdp::ControlledHopOnMDP) = mdp.actions
POMDPs.actions(mdp::ControlledHopOnMDP, s::ControlledHopOnStateAugmented) = mdp.actions
POMDPs.n_actions(mdp::ControlledHopOnMDP) = length(mdp.actions)
POMDPs.discount(mdp::ControlledHopOnMDP) = 1.0
POMDPs.actionindex(mdp::ControlledHopOnMDP, a::HopOnAction) = a.action_idx


"""
For constrained flight, terminal state when horizon 0
"""
function POMDPs.isterminal(mdp::ControlledHopOnMDP, s::ControlledHopOnStateAugmented)

    # If horizon is 0, then terminal
    # s.rel_uavstate.x == Inf is an explicit terminal state that arises from terminal reward
    return s.horizon <= 0

end

"""
Generate next constrained flight state based on UAV dynamics or abort.
"""
function POMDPs.generate_sr(mdp::ControlledHopOnMDP, s::ControlledHopOnStateAugmented, 
                     a::HopOnAction, rng::RNG=Base.GLOBAL_RNG) where {RNG <: AbstractRNG}

    cost = mdp.energy_time_alpha*mdp.params.cost_params.TIME_COEFFICIENT*mdp.params.time_params.MDP_TIMESTEP

    # Depending on action, propagate UAV dynamics or abort
    if a.uavaction != nothing
        new_uavstate = next_state(mdp.dynamics, s.rel_uavstate, a.uavaction, rng)
        cost += (1.0 - mdp.energy_time_alpha)*dynamics_cost(mdp.dynamics, s.rel_uavstate, new_uavstate)

        if s.horizon == 1
            # If unable to complete hop, penalize
            if mdp.terminal_costs_set
                curr_pos = get_position(new_uavstate)
                curr_speed = get_speed(new_uavstate)
                if point_norm(curr_pos) > mdp.params.time_params.MDP_TIMESTEP*mdp.params.scale_params.HOP_DISTANCE_THRESHOLD || curr_speed > mdp.params.scale_params.XYDOT_HOP_THRESH
                    cost += mdp.no_hop_penalty
                end
            end
        end

        return ControlledHopOnStateAugmented(new_uavstate, s.horizon-1), -cost
    elseif a.control_transfer == true
        # TODO : No point setting x to Inf
        # Lookup horizon-based cost and add here before returning terminal state
        if s.horizon <= mdp.params.time_params.HORIZON_LIM # For the out-horizon case where horizon nominally set to 0
            cost += mdp.horizon_abort_penalty[s.horizon]
        else
            cost = Inf
        end

        return ControlledHopOnStateAugmented(s.rel_uavstate, -1), -cost
    else
        throw(ArgumentError("Invalid action specified!"))
    end
end

# Explicit transition deprecated
# function transition(mdp::ControlledMultiRotorHopOnMDP, s::ControlledHopOnStateAugmented, a::HopOnAction)

#     if a.uavaction != nothing

#         sigma_uavstates, sigma_probs = sigma_point_states_weights(mdp.dynamics, s.rel_uavstate, a.uavaction)
#         n_nbrs = length(sigma_uavstates)
#         nbr_states = Vector{ControlledHopOnStateAugmented}(n_nbrs)

#         for i = 1:n_nbrs
#             nbr_states[i] = ControlledHopOnStateAugmented(sigma_uavstates[i], s.horizon-1)
#         end

#         return SparseCat(nbr_states, sigma_probs)
#     elseif a.control_transfer == true
#         return SparseCat([ControlledHopOnStateAugmented(s.rel_uavstate, -1)],[1.0])
#     else
#         throw(ArgumentError("Invalid action specified!"))
#     end
# end

function reward(mdp::ControlledHopOnMDP, s::ControlledHopOnStateAugmented, a::HopOnAction, sp::ControlledHopOnStateAugmented)

    cost = mdp.energy_time_alpha*mdp.params.cost_params.TIME_COEFFICIENT*mdp.params.time_params.MDP_TIMESTEP

    if a.uavaction != nothing
        cost += dynamics_cost(mdp.dynamics, s.rel_uavstate, sp.rel_uavstate)
        if sp.horizon == 0
            if mdp.terminal_costs_set
                curr_pos = get_position(sp)
                curr_speed = get_speed(sp)
                if point_norm(curr_pos) > mdp.params.time_params.MDP_TIMESTEP*mdp.params.scale_params.HOP_DISTANCE_THRESHOLD || curr_speed > mdp.params.scale_params.XYDOT_HOP_THRESH
                    cost += mdp.no_hop_penalty
                end
            end
        end
    else
        if s.horizon <= mdp.params.time_params.HORIZON_LIM
            cost += mdp.horizon_abort_penalty[s.horizon]
        else
            cost = Inf
        end
    end

    # For the other action, no additional cost accrued
    return -cost
end

"""
    generate_time_to_finish_dist(curr_time_to_fin::Float64, rng::RNG=Base.GLOBAL_RNG) where {RNG <: AbstractRNG}

Generate a distribution over horizon values left for the uncontrolled problem to terminate, by sampling a number of finish times
and binning them into histogram horizon values.
"""
function generate_time_to_finish_dist(curr_time_to_fin::Float64, params::Parameters, rng::RNG=Base.GLOBAL_RNG) where {RNG <: AbstractRNG}

    time_to_finish = Distributions.Normal(curr_time_to_fin, params.time_params.CAR_TIME_STD)
    time_to_finish_prob = zeros(params.time_params.HORIZON_LIM+2)

    for j = 1:params.time_params.MC_TIME_NUMSAMPLES
        tval = rand(rng, time_to_finish)/params.time_params.MDP_TIMESTEP
        if tval >= params.time_params.HORIZON_LIM
            time_to_finish_prob[end] += 1.0
        else
            low = convert(Int64,floor(tval))
            high = convert(Int64,ceil(tval))
            low_wt = tval - floor(tval)
            time_to_finish_prob[max(1,low+1)] += low_wt
            time_to_finish_prob[max(1,high+1)] += 1.0 - low_wt
        end
    end

    # Ensure that the distribution is valid
    @assert sum(time_to_finish_prob) > 0.0
    time_to_finish_prob /= sum(time_to_finish_prob)

    return time_to_finish_prob
end


"""
A general policy object for representing the finite horizon policies (constrained flight and hitchhiking)
"""
struct PartialControlHopOnOffPolicy{UA <: UAVAction}
    in_horizon_policy::LocalApproximationValueIterationPolicy
    out_horizon_policy::LocalApproximationValueIterationPolicy
    action_map::Union{Vector{HopOnAction{UA}}, Vector{HopOffAction}}
end



"""
    hopon_policy_action(policy::PartialControlHopOnOffPolicy, rel_uavstate::US, 
                        curr_time_to_fin::Float64, rng::RNG=Base.GLOBAL_RNG) where {US <: UAVState,RNG <: AbstractRNG}

Take an action according to the constrained flight policy, given a distribution of the remaining. This implements
the equation from the paper on Collision Avoidance using Partially Controlled MDPs.
"""
function hopon_policy_action(policy::PartialControlHopOnOffPolicy, rel_uavstate::US, 
                             time_to_finish_prob::Vector{Float64}, rng::RNG=Base.GLOBAL_RNG) where {US <: UAVState,RNG <: AbstractRNG}
    
    mdp = policy.in_horizon_policy.mdp

    action_values = zeros(n_actions(mdp))

    for a in actions(mdp)
        iaction = actionindex(mdp,a)

        # Horizon 0 value same for all actions - ignore
        for hor = 1:mdp.params.time_params.HORIZON_LIM
            time_prob = time_to_finish_prob[hor+1]

            if time_prob > 0.0
                aug_inhor_state = ControlledHopOnStateAugmented(rel_uavstate,hor)
                action_values[iaction] += time_prob*action_value(policy.in_horizon_policy, aug_inhor_state, a)
            end
        end

        time_prob = time_to_finish_prob[end]
        if time_prob > 0.0
            aug_outhor_state = ControlledHopOnStateAugmented(rel_uavstate,mdp.params.time_params.HORIZON_LIM+1)
            action_values[iaction] += time_prob*action_value(policy.out_horizon_policy,aug_outhor_state,a)
        end
    end

    best_action_idx = argmax(action_values)
    best_action = policy.in_horizon_policy.action_map[best_action_idx]

    return best_action

end

"""
    hopon_policy_action(policy::PartialControlHopOnOffPolicy, rel_uavstate::US, 
                        curr_time_to_fin::Float64, rng::RNG=Base.GLOBAL_RNG) where {US <: UAVState,RNG <: AbstractRNG}

Take an action according to the constrained flight policy, given a point estimate of the arrival time. This constructs a 
distribution over the remaining time and calls the corresponding hopon_policy_action method.
"""
function hopon_policy_action(policy::PartialControlHopOnOffPolicy, params::Parameters, rel_uavstate::US, 
                                curr_time_to_fin::Float64, rng::RNG=Base.GLOBAL_RNG) where {US <: UAVState,RNG <: AbstractRNG}
    # Set up the time to finish samples
    time_to_finish_prob = generate_time_to_finish_dist(curr_time_to_fin, params, rng)
    return hopon_policy_action(policy,rel_uavstate,time_to_finish_prob,rng)
end


"""
    hopoff_policy_action(policy::PartialControlHopOnOffPolicy, time_to_finish_prob::Vector{Float64}, rng::RNG=Base.GLOBAL_RNG) where {RNG <: AbstractRNG}

Take an action according to the hitchiking policy. This is typically simple, i.e. STAY till close to the horizon limit, then hop off
"""
function hopoff_policy_action(policy::PartialControlHopOnOffPolicy, time_to_finish_prob::Vector{Float64}, rng::RNG=Base.GLOBAL_RNG) where {RNG <: AbstractRNG}
    
    mdp = policy.in_horizon_policy.mdp

    action_values = zeros(n_actions(mdp))

    for a in actions(mdp)
        iaction = actionindex(mdp,a)

        # Horizon 0 value same for all actions - ignore
        for hor = 1:mdp.params.time_params.HORIZON_LIM
            time_prob = time_to_finish_prob[hor+1]

            if time_prob > 0.0
                aug_inhor_state = HopOffStateAugmented(true,hor)
                action_values[iaction] += time_prob*action_value(policy.in_horizon_policy, aug_inhor_state, a)
            end
        end

        time_prob = time_to_finish_prob[end]
        if time_prob > 0.0
            aug_outhor_state = HopOffStateAugmented(true,mdp.params.time_params.HORIZON_LIM+1)
            action_values[iaction] += time_prob*action_value(policy.out_horizon_policy,aug_outhor_state,a)
        end
    end

    best_action_idx = argmax(action_values)
    best_action = policy.in_horizon_policy.action_map[best_action_idx]

    # Could be either hop on or hop off action
    return best_action
end


"""
    hopoff_policy_action(policy::PartialControlHopOnOffPolicy, curr_time_to_fin::Float64,rng::RNG=Base.GLOBAL_RNG) where {RNG <: AbstractRNG}

Take an action according to the hitchiking policy, by generating a time to finish distribution and calling the corresponding method for a distribution.
"""
function hopoff_policy_action(policy::PartialControlHopOnOffPolicy, params::Parameters, curr_time_to_fin::Float64, rng::RNG=Base.GLOBAL_RNG) where {RNG <: AbstractRNG}

    # Set up the time to finish samples
    time_to_finish_prob = generate_time_to_finish_dist(curr_time_to_fin, params, rng)
    return hopoff_policy_action(policy, time_to_finish_prob, rng)
end

# TODO: Deprecated
# function get_acc_along_axis_outhordist(axis_pos_diff::Float64, axis_dot::Float64)

#     if axis_pos_diff < 1.05*XY_LIM
#         # Just take it to rest
#         acc = min(abs(axis_dot/MDP_TIMESTEP), ACCELERATION_LIM)
#         acc = copysign(acc, -axis_dot)
#     else
#         if sign(axis_pos_diff) == sign(axis_dot)
#             acc = copysign(ACCELERATION_LIM, -axis_dot)
#         else
#             if abs(axis_dot) >= XYDOT_LIM
#                 acc = 0.0
#             else
#                 acc = copysign(ACCELERATION_LIM, axis_dot)
#             end
#         end
#     end
#     return acc
# end

# function outhor_outdist_action(rel_uavstate::MultiRotorUAVState)
#     acc_x = get_acc_along_axis_outhordist(rel_uavstate.x, rel_uavstate.xdot)
#     acc_y = get_acc_along_axis_outhordist(rel_uavstate.y, rel_uavstate.ydot)
#     return HopOnAction(-1, MultiRotorUAVAction(acc_x, acc_y), nothing)
# end


# function get_acc_along_axis_unconstrained(axis_pos_diff::Float64, axis_dot::Float64)

#     if sign(axis_pos_diff) == sign(axis_dot)
#         acc = copysign(ACCELERATION_LIM, -axis_dot)
#     else
#         if axis_pos_diff > 10*XYDOT_LIM*MDP_TIMESTEP
#             acc = copysign(ACCELERATION_LIM, axis_dot)
#         else
#             acc_untrunc = min(abs(-2*(axis_pos_diff + axis_dot*MDP_TIMESTEP)/(MDP_TIMESTEP^2)), ACCELERATION_LIM)
#             acc = copysign(acc_untrunc, axis_dot)
#         end
#     end
#     return acc
# end


# function unconstrained_flight_action(rel_uavstate::MultiRotorUAVState)
#     acc_x = get_acc_along_axis_unconstrained(rel_uavstate.x, rel_uavstate.xdot)
#     acc_y = get_acc_along_axis_unconstrained(rel_uavstate.y, rel_uavstate.ydot)
#     return MultiRotorUAVAction(acc_x, acc_y)
# end


# State conversion functions
# IMP! They must be specific to the MDP type because it must be explicitly unrolled into a vector
function POMDPs.convert_s(::Type{V}, s::ControlledHopOnStateAugmented{MultiRotorUAVState}, 
                          mdp::ControlledHopOnMDP{MultiRotorUAVState,MultiRotorUAVAction,MultiRotorUAVDynamicsModel}) where V <: AbstractVector{Float64}
  v = SVector{5,Float64}(s.rel_uavstate.x, s.rel_uavstate.y, s.rel_uavstate.xdot, s.rel_uavstate.ydot, convert(Float64,s.horizon))
  return v
end

function POMDPs.convert_s(::Type{ControlledHopOnStateAugmented{MultiRotorUAVState}}, v::AbstractVector{Float64},
                          mdp::ControlledHopOnMDP{MultiRotorUAVState,MultiRotorUAVAction,MultiRotorUAVDynamicsModel})
  s = ControlledHopOnStateAugmented(MultiRotorUAVState(v[1],v[2],v[3],v[4]), convert(Int64,v[5]))
  return s
end


################################## UNCONSTRAINED FLIGHT MDP ##################################
"""
A FlightAction is just a valid UAV dynamics action with some index.
"""
struct FlightAction{UA <: UAVAction}
    action_idx::Int
    uav_action::UA
end

"""
The unconstrained flight mdp represents the last stage of the agent's journey, from the last hop off point
and going to the goal position (i.e. relative position 0)
"""
mutable struct UnconstrainedFlightMDP{US <: UAVState, UA <: UAVAction, UDM <: UAVDynamicsModel} <: POMDPs.MDP{US, FlightAction{UA}}
    dynamics::UDM
    discount::Float64
    actions::Vector{FlightAction{UA}}
    energy_time_alpha::Float64
    params::Parameters
end

POMDPs.actions(mdp::UnconstrainedFlightMDP) = mdp.actions
POMDPs.actions(mdp::UnconstrainedFlightMDP, s::US) where {US<:UAVState} = mdp.actions
POMDPs.n_actions(mdp::UnconstrainedFlightMDP) = length(mdp.actions)
POMDPs.discount(mdp::UnconstrainedFlightMDP) = mdp.discount # NOTE - Needs to be < 1.0 as infinite horizon problem

"""
    UnconstrainedFlightMDP{UA}(dynamics::UDM, discount::Float64, energy_time_alpha::Float64=0.5) where {US <: UAVState, UA <: UAVAction, UDM <: UAVDynamicsModel}

Constructor for Flight MDP requires types of state, action, and dynamics.
"""
function UnconstrainedFlightMDP{US, UA}(dynamics::UDM, discount::Float64, energy_time_alpha::Float64=0.5) where {US <: UAVState, UA <: UAVAction, UDM <: UAVDynamicsModel}

    uav_dynamics_actions = get_uav_dynamics_actions(dynamics)
    uav_flight_actions = Vector{FlightAction{UA}}(undef,0)
    idx::Int64 = 1

    for (i,uavda) in enumerate(uav_dynamics_actions)
        push!(uav_flight_actions, FlightAction(i,uavda))
    end

    return UnconstrainedFlightMDP{US, UA, UDM}(dynamics, discount, uav_flight_actions, energy_time_alpha, dynamics.params)
end


function POMDPs.actionindex(mdp::UnconstrainedFlightMDP, a::FlightAction)
    return a.action_idx
end


function POMDPs.isterminal(mdp::UnconstrainedFlightMDP, s::US) where {US <: UAVState}
    curr_pos_norm = point_norm(get_position(s))
    curr_speed = get_speed(s)

    if (curr_pos_norm < (mdp.params.time_params.MDP_TIMESTEP*mdp.params.scale_params.HOP_DISTANCE_THRESHOLD)) && 
        (curr_speed < mdp.params.scale_params.XYDOT_HOP_THRESH)
        return true
    end
    return false
    # return (2.3 < 3.4)
end


function POMDPs.generate_sr(mdp::UnconstrainedFlightMDP, s::US, a::FlightAction, rng::RNG=Base.GLOBAL_RNG) where {US <: UAVState, RNG <: AbstractRNG}
    
    cost = mdp.energy_time_alpha*mdp.params.cost_params.TIME_COEFFICIENT*mdp.params.time_params.MDP_TIMESTEP

    new_uavstate = next_state(mdp.dynamics, s, a.uav_action, rng)
    
    cost += (1.0-mdp.energy_time_alpha)*dynamics_cost(mdp.dynamics, s, new_uavstate)

    # TODO : Why is this needed? Debug
    if isterminal(mdp, new_uavstate)
        cost -= mdp.params.cost_params.FLIGHT_REACH_REWARD
    end

    return new_uavstate, -cost
end

function POMDPs.reward(mdp::UnconstrainedFlightMDP, s::US, a::FlightAction, sp::US) where {US <: UAVState}

    cost = mdp.energy_time_alpha*mdp.params.cost_params.TIME_COEFFICIENT*mdp.params.time_params.MDP_TIMESTEP + 
            (1.0-mdp.energy_time_alpha)*dynamics_cost(mdp.dynamics, s, sp.rel_uavstate)
    return -cost 
end


# State conversion functions
# Again, these are specific to multirotor flight MDP
function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, s::MultiRotorUAVState, 
                          mdp::Any)
  v = SVector{4,Float64}(s.x, s.y, s.xdot, s.ydot)
  return v
end

function POMDPs.convert_s(::Type{MultiRotorUAVState}, v::AbstractVector{Float64}, 
                          mdp::Any)
  s = MultiRotorUAVState(v[1],v[2],v[3],v[4])
  return s
end