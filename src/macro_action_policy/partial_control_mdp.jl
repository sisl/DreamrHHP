importall POMDPs
# This is the controlled state space of the UAV when it is planning
# to track the car and hop on to it. It must try to reach the goal 
# position at or before the car reaches it
mutable struct ControlledHopOnStateAugmented
    # States pertaining to this subproblem
    rel_uavstate::US where {US <: UAVState}# RELATIVE to goal
    horizon::Int64 # POMDPs.jl has no explicit interface for finite horizon problems
end

# For now, this just has the horizon as the dynamics depend on that of the car
# We want the drone to stay on till the car is close to the transfer point
mutable struct HopOffStateAugmented
    oncar::Bool
    horizon::Int64
end


# NOTE
mutable struct HopOnAction
    action_idx::Int64
    uavaction::Union{Void,UA} where {UA <: UAVAction}
    control_transfer::Union{Void,Bool}
end

mutable struct HopOffAction
    action_idx::Int64
    hopaction::HOP_ACTION
end


######################## HOPOFF MDP ##########################
mutable struct HopOffMDP <: POMDPs.MDP{HopOffStateAugmented,HopOffAction}
    actions::Vector{HopOffAction}
    terminal_costs_set::Bool
end

function HopOffMDP()
    hopoff_actions = [HopOffAction(1,STAY), HopOffAction(2,HOPOFF)]
    return HopOffMDP(hopoff_actions,true)
end

actions(mdp::HopOffMDP) = mdp.actions
actions(mdp::HopOffMDP, s::HopOffStateAugmented) = mdp.actions
n_actions(mdp::HopOffMDP) = length(mdp.actions)
discount(mdp::HopOffMDP) = 1.0

function action_index(mdp::HopOffMDP, a::HopOffAction)
    return a.action_idx
end

function isterminal(mdp::HopOffMDP, s::HopOffStateAugmented)
    return (s.oncar == false || s.horizon == 0)
end

function transition(mdp::HopOffMDP, s::HopOffStateAugmented, a::HopOffAction)

    if a.hopaction == STAY
        return SparseCat([HopOffStateAugmented(true,s.horizon-1)],[1.0])
    else
        return SparseCat([HopOffStateAugmented(false,s.horizon-1)],[1.0])
    end
end

function reward(mdp::HopOffMDP, s::HopOffStateAugmented, a::HopOffAction, sp::HopOffStateAugmented)
    if mdp.terminal_costs_set
        if (sp.horizon == 0 && sp.oncar == true) || (sp.horizon > 0 && sp.oncar == false)
            return -NO_HOPOFF_PENALTY
        end
    end
    return -(TIME_COEFFICIENT*MDP_TIMESTEP)
end

# TODO : Need a finite horizon partially controlled MDP policy
# For now, use horizon as part of state
mutable struct PartialControlHopOnOffPolicy
    in_horizon_policy::LocalApproximationValueIterationPolicy
    out_horizon_policy::LocalApproximationValueIterationPolicy
    action_map::Vector
end

function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, s::HopOffStateAugmented, mdp::HopOffMDP)
  v = SVector{2,Float64}(convert(Float64,s.oncar), convert(Float64,s.horizon))
  return v
end

function POMDPs.convert_s(::Type{HopOffStateAugmented}, v::AbstractVector{Float64}, mdp::HopOffMDP)
  s = HopOffStateAugmented(convert(Bool,v[1]), convert(Int64,v[2]))
  return s
end

######################## MULTI ROTOR MDP ##########################
mutable struct ControlledMultiRotorHopOnMDP <: POMDPs.MDP{ControlledHopOnStateAugmented,HopOnAction}
    dynamics::MultiRotorUAVDynamicsModel
    actions::Vector{HopOnAction}
    horizon_abort_penalty::Vector{Float64}
    terminal_costs_set::Bool
    no_hop_penalty::Float64
end

function ControlledMultiRotorHopOnMDP(dynamics::MultiRotorUAVDynamicsModel)

    multi_rotor_hopon_actions = Vector{HopOnAction}()
    idx::Int64 = 1

    acc_vals = linspace(-ACCELERATION_LIM,ACCELERATION_LIM,ACCELERATION_NUMVALS)
    
    for xddot in acc_vals
        for yddot in acc_vals
            push!(multi_rotor_hopon_actions,HopOnAction(idx,MultiRotorUAVAction(xddot,yddot),nothing))
            idx+=1
        end
    end
    push!(multi_rotor_hopon_actions, HopOnAction(idx,nothing,true))

    return ControlledMultiRotorHopOnMDP(dynamics,multi_rotor_hopon_actions,Inf*ones(HORIZON_LIM),true,Inf)
end


actions(mdp::ControlledMultiRotorHopOnMDP) = mdp.actions
actions(mdp::ControlledMultiRotorHopOnMDP, s::ControlledHopOnStateAugmented) = mdp.actions
n_actions(mdp::ControlledMultiRotorHopOnMDP) = length(mdp.actions)
discount(mdp::ControlledMultiRotorHopOnMDP) = 1.0


function action_index(mdp::ControlledMultiRotorHopOnMDP, a::HopOnAction)
    return a.action_idx
end

# TODO - Need isterminal and a method to get immediate cost for s_c
function isterminal(mdp::ControlledMultiRotorHopOnMDP, s::ControlledHopOnStateAugmented)

    # If horizon is 0, then terminal
    # s.rel_uavstate.x == Inf is an explicit terminal state that arises from terminal reward
    return s.horizon <= 0

end


### TRANSITION ###
# Monte Carlo, will also need to do sigma point sampling later
# NOTE - This is for training, for actual will append horizon
# to create multiple augmented states.
function generate_sr(mdp::ControlledMultiRotorHopOnMDP, s::ControlledHopOnStateAugmented, 
                     a::HopOnAction, rng::RNG=Base.GLOBAL_RNG) where {RNG <: AbstractRNG}

    cost = TIME_COEFFICIENT*MDP_TIMESTEP

    # Can assume control_transfer is false

    # Depending on action, do various things
    if a.uavaction != nothing
        new_uavstate = next_state(mdp.dynamics, s.rel_uavstate, a.uavaction, rng)
        cost += dynamics_cost(mdp.dynamics, s.rel_uavstate, new_uavstate)

        if s.horizon == 1
            # Add terminal reward if appropriate
            if mdp.terminal_costs_set
                curr_pos = Point(new_uavstate.x, new_uavstate.y)
                curr_speed = sqrt(new_uavstate.xdot^2 + new_uavstate.ydot^2)
                if point_norm(curr_pos) > MDP_TIMESTEP*HOP_DISTANCE_THRESHOLD || curr_speed > XYDOT_HOP_THRESH
                    cost += mdp.no_hop_penalty
                end
            end
        end

        return ControlledHopOnStateAugmented(new_uavstate, s.horizon-1), -cost
    elseif a.control_transfer == true
        # TODO : No point setting x to Inf
        # Lookup horizon-based cost and add here before returning terminal state
        if s.horizon <= HORIZON_LIM # For the out-horizon case where horizon nominally set to 0
            cost += mdp.horizon_abort_penalty[s.horizon]
        else
            cost = Inf
        end

        return ControlledHopOnStateAugmented(s.rel_uavstate, -1), -cost
    else
        throw(ArgumentError("Invalid action specified!"))
    end
end

function transition(mdp::ControlledMultiRotorHopOnMDP, s::ControlledHopOnStateAugmented, a::HopOnAction)

    if a.uavaction != nothing

        sigma_uavstates, sigma_probs = sigma_point_states_weights(mdp.dynamics, s.rel_uavstate, a.uavaction)
        n_nbrs = length(sigma_uavstates)
        nbr_states = Vector{ControlledHopOnStateAugmented}(n_nbrs)

        for i = 1:n_nbrs
            nbr_states[i] = ControlledHopOnStateAugmented(sigma_uavstates[i], s.horizon-1)
        end

        return SparseCat(nbr_states, sigma_probs)
    elseif a.control_transfer == true
        return SparseCat([ControlledHopOnStateAugmented(s.rel_uavstate, -1)],[1.0])
    else
        throw(ArgumentError("Invalid action specified!"))
    end
end

function reward(mdp::ControlledMultiRotorHopOnMDP, s::ControlledHopOnStateAugmented, a::HopOnAction, sp::ControlledHopOnStateAugmented)

    cost = TIME_COEFFICIENT*MDP_TIMESTEP

    if a.uavaction != nothing
        cost += dynamics_cost(mdp.dynamics, s.rel_uavstate, sp.rel_uavstate)
        if sp.horizon == 0
            if mdp.terminal_costs_set
                curr_pos = Point(sp.rel_uavstate.x, sp.rel_uavstate.y)
                curr_speed = sqrt(new_uavstate.xdot^2 + new_uavstate.ydot^2)
                if point_norm(curr_pos) > MDP_TIMESTEP*HOP_DISTANCE_THRESHOLD || curr_speed > XYDOT_HOP_THRESH
                    cost += mdp.no_hop_penalty
                end
            end
        end
    else
        if s.horizon <= HORIZON_LIM
            cost += mdp.horizon_abort_penalty[s.horizon]
        else
            cost = Inf
        end
    end

    # For the other action, no additional cost accrued
    return -cost
end

function generate_time_to_finish_dist(curr_time_to_fin::Float64, rng::RNG=Base.GLOBAL_RNG) where {RNG <: AbstractRNG}

    time_to_finish = Distributions.Normal(curr_time_to_fin, CAR_TIME_STD)
    time_to_finish_prob = zeros(HORIZON_LIM+2)

    for j = 1:MC_TIME_NUMSAMPLES
        tval = rand(rng, time_to_finish)/MDP_TIMESTEP
        if tval >= HORIZON_LIM
            time_to_finish_prob[end] += 1.0
        else
            # Tricky part - don't round off but rather allocate
            low = Int64(floor(tval))
            high = Int64(ceil(tval))
            low_wt = tval - floor(tval)
            time_to_finish_prob[max(1,low+1)] += low_wt
            time_to_finish_prob[max(1,high+1)] += 1.0 - low_wt
        end
    end

    @assert sum(time_to_finish_prob) > 0.0
    time_to_finish_prob /= sum(time_to_finish_prob)

    return time_to_finish_prob
end


# IMP - Actual hopon to be decided by upper layer
function hopon_policy_action(policy::PartialControlHopOnOffPolicy, rel_uavstate::US, 
                                curr_time_to_fin::Float64, rng::RNG=Base.GLOBAL_RNG) where {US <: UAVState,RNG <: AbstractRNG}

    # Set up the time to finish samples
    time_to_finish_prob = generate_time_to_finish_dist(curr_time_to_fin,rng)
    mdp = policy.in_horizon_policy.mdp

    action_values = zeros(n_actions(mdp))

    for a in iterator(actions(mdp))
        iaction = action_index(mdp,a)

        # Horizon 0 value same for all actions - ignore
        for hor = 1:HORIZON_LIM
            time_prob = time_to_finish_prob[hor+1]

            if time_prob > 0.0
                aug_inhor_state = ControlledHopOnStateAugmented(rel_uavstate,hor)
                action_values[iaction] += time_prob*action_value(policy.in_horizon_policy, aug_inhor_state, a)
            end
        end

        time_prob = time_to_finish_prob[end]
        if time_prob > 0.0
            aug_outhor_state = ControlledHopOnStateAugmented(rel_uavstate,HORIZON_LIM+1)
            action_values[iaction] += time_prob*action_value(policy.out_horizon_policy,aug_outhor_state,a)
        end
    end

    best_action_idx = indmax(action_values)
    best_action = policy.in_horizon_policy.action_map[best_action_idx]

    # Could be either hop on or hop off action
    return best_action
end

function hopoff_policy_action(policy::PartialControlHopOnOffPolicy, curr_time_to_fin::Float64,rng::RNG=Base.GLOBAL_RNG) where {RNG <: AbstractRNG}

    # Set up the time to finish samples
    time_to_finish_prob = generate_time_to_finish_dist(curr_time_to_fin)
    mdp = policy.in_horizon_policy.mdp

    action_values = zeros(n_actions(mdp))

    for a in iterator(actions(mdp))
        iaction = action_index(mdp,a)

        # Horizon 0 value same for all actions - ignore
        for hor = 1:HORIZON_LIM
            time_prob = time_to_finish_prob[hor+1]

            if time_prob > 0.0
                aug_inhor_state = HopOffStateAugmented(true,hor)
                action_values[iaction] += time_prob*action_value(policy.in_horizon_policy, aug_inhor_state, a)
            end
        end

        time_prob = time_to_finish_prob[end]
        if time_prob > 0.0
            aug_outhor_state = HopOffStateAugmented(true,HORIZON_LIM+1)
            action_values[iaction] += time_prob*action_value(policy.out_horizon_policy,aug_outhor_state,a)
        end
    end

    best_action_idx = indmax(action_values)
    best_action = policy.in_horizon_policy.action_map[best_action_idx]

    # Could be either hop on or hop off action
    return best_action
end


function get_acc_along_axis_outhordist(axis_pos_diff::Float64, axis_dot::Float64)

    if axis_pos_diff < 1.05*XY_LIM
        # Just take it to rest
        acc = min(abs(axis_dot/MDP_TIMESTEP), ACCELERATION_LIM)
        acc = copysign(acc, -axis_dot)
    else
        if sign(axis_pos_diff) == sign(axis_dot)
            acc = copysign(ACCELERATION_LIM, -axis_dot)
        else
            if abs(axis_dot) >= XYDOT_LIM
                acc = 0.0
            else
                acc = copysign(ACCELERATION_LIM, axis_dot)
            end
        end
    end
    return acc
end

function outhor_outdist_action(rel_uavstate::MultiRotorUAVState)
    acc_x = get_acc_along_axis_outhordist(rel_uavstate.x, rel_uavstate.xdot)
    acc_y = get_acc_along_axis_outhordist(rel_uavstate.y, rel_uavstate.ydot)
    return HopOnAction(-1, MultiRotorUAVAction(acc_x, acc_y), nothing)
end


function get_acc_along_axis_unconstrained(axis_pos_diff::Float64, axis_dot::Float64)

    if sign(axis_pos_diff) == sign(axis_dot)
        acc = copysign(ACCELERATION_LIM, -axis_dot)
    else
        if axis_pos_diff > 10*XYDOT_LIM*MDP_TIMESTEP
            acc = copysign(ACCELERATION_LIM, axis_dot)
        else
            acc_untrunc = min(abs(-2*(axis_pos_diff + axis_dot*MDP_TIMESTEP)/(MDP_TIMESTEP^2)), ACCELERATION_LIM)
            acc = copysign(acc_untrunc, axis_dot)
        end
    end
    return acc
end


function unconstrained_flight_action(rel_uavstate::MultiRotorUAVState)
    acc_x = get_acc_along_axis_unconstrained(rel_uavstate.x, rel_uavstate.xdot)
    acc_y = get_acc_along_axis_unconstrained(rel_uavstate.y, rel_uavstate.ydot)
    return MultiRotorUAVAction(acc_x, acc_y)
end


# State conversion functions
function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, s::ControlledHopOnStateAugmented, mdp::ControlledMultiRotorHopOnMDP)
  v = SVector{5,Float64}(s.rel_uavstate.x, s.rel_uavstate.y, s.rel_uavstate.xdot, s.rel_uavstate.ydot,convert(Float64,s.horizon))
  return v
end

function POMDPs.convert_s(::Type{ControlledHopOnStateAugmented}, v::AbstractVector{Float64}, mdp::ControlledMultiRotorHopOnMDP)
  s = ControlledHopOnStateAugmented(MultiRotorUAVState(v[1],v[2],v[3],v[4]),convert(Int64,v[5]))
  return s
end




###################### UNCONSTRAINED FLIGHT MDP ######################
struct FlightAction{UA <: UAVAction}
    action_idx::Int
    uav_action::UA
end

mutable struct UnconstrainedFlightMDP{US <: UAVState, FA <: FlightAction, UDM <: UAVDynamicsModel} <: POMDPs.MDP{US, FA}
    dynamics::UDM
    discount::Float64
    actions::Vector{FA}
end

actions(mdp::UnconstrainedFlightMDP) = mdp.actions
actions(mdp::UnconstrainedFlightMDP, s::US) where {US<:UAVState} = mdp.actions
n_actions(mdp::UnconstrainedFlightMDP) = length(mdp.actions)
discount(mdp::UnconstrainedFlightMDP) = mdp.discount # NOTE - Needs to be < 1.0 as infinite horizon problem


function UnconstrainedFlightMDP(dynamics::MultiRotorUAVDynamicsModel, discount::Float64)

    multi_rotor_flight_actions = Vector{FlightAction{MultiRotorUAVAction}}()
    idx::Int64 = 1

    acc_vals = linspace(-ACCELERATION_LIM,ACCELERATION_LIM,ACCELERATION_NUMVALS)

    for xddot in acc_vals
        for yddot in acc_vals
            push!(multi_rotor_flight_actions,FlightAction(idx,MultiRotorUAVAction(xddot,yddot)))
            idx+=1
        end
    end

    println(multi_rotor_flight_actions)

    return UnconstrainedFlightMDP{MultiRotorUAVState, FlightAction{MultiRotorUAVAction}, typeof(dynamics)}(dynamics, discount, multi_rotor_flight_actions)
end

function action_index(mdp::UnconstrainedFlightMDP, a::FlightAction)
    return a.action_idx
end

# TODO : Assume that all UAVStates have x and y 
function isterminal(mdp::UnconstrainedFlightMDP, s::US) where {US <: UAVState}
    curr_pos = Point(s.x, s.y)
    curr_speed = sqrt(s.xdot^2 + s.ydot^2)
    return point_norm(curr_pos) < MDP_TIMESTEP*HOP_DISTANCE_THRESHOLD && curr_speed < XYDOT_HOP_THRESH
end

function generate_sr(mdp::UnconstrainedFlightMDP, s::US, a::FlightAction, rng::RNG=Base.GLOBAL_RNG) where {US <: UAVState, RNG <: AbstractRNG}
    
    cost = TIME_COEFFICIENT*MDP_TIMESTEP

    new_uavstate = next_state(mdp.dynamics, s, a.uav_action, rng)
    cost += dynamics_cost(mdp.dynamics, s, new_uavstate)

    if isterminal(mdp, new_uavstate)
        cost -= FLIGHT_REACH_REWARD
    end

    return new_uavstate, -cost
end

function reward(mdp::UnconstrainedFlightMDP, s::US, a::FlightAction, sp::US) where {US <: UAVState}

    cost = TIME_COEFFICIENT*MDP_TIMESTEP
    cost += dynamics_cost(mdp.dynamics, s, sp.rel_uavstate)
    return -cost 

end


# State conversion functions
function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, s::MultiRotorUAVState, mdp::Any)
  v = SVector{4,Float64}(s.x, s.y, s.xdot, s.ydot)
  return v
end

function POMDPs.convert_s(::Type{MultiRotorUAVState}, v::AbstractVector{Float64}, mdp::Any)
  s = MultiRotorUAVState(v[1],v[2],v[3],v[4])
  return s
end