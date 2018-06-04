importall POMDPs
# This is the controlled state space of the UAV when it is planning
# to track the car and hop on to it. It must try to reach the goal 
# position at or before the car reaches it
mutable struct ControlledHopOnStateAugmented
    # States pertaining to this subproblem
    rel_uavstate::US where {US <: UAVState}# RELATIVE to goal
    control_transfer::Bool
    horizon::Int64 # POMDPs.jl has no explicit interface for finite horizon problems
end

mutable struct ControlledHopOnState
    rel_uavstate::US where {US<:UAVState} # RELATIVE to goal
    control_transfer::Bool
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
end

function HopOffMDP()
    hopoff_actions = [HopOffAction(1,STAY), HopOffAction(2,HOPOFF)]
    return HopOffMDP(hopoff_actions)
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

function terminalreward(mdp::HopOffMDP, s::HopOffStateAugmented)
    if !isterminal(mdp,s)
        return 0.0
    end

    if s.oncar == true
        # Still on car at zero horizon
        return -HOP_REWARD
    end

    # Now not on car, but if not at end of horizon, penalize
    if s.horizon > 0
        return -HOP_REWARD
    end

    # Now we know it is off car at horizon 0, so reward
    # NOTE : This means that the moment the car has more probability
    # to finish at the next timestep than anything beyond that, it will hop off
    return HOP_REWARD
end

function transition(mdp::HopOffMDP, s::HopOffStateAugmented, a::HopOffAction)

    if a.hopaction == STAY
        return SparseCat([HopOffStateAugmented(true,s.horizon-1)],[1.0])
    else
        return SparseCat([HopOffStateAugmented(false,s.horizon-1)],[1.0])
    end
end

function reward(mdp::HopOffMDP, s::HopOffStateAugmented, a::HopOffAction, sp::HopOffStateAugmented)
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

    return ControlledMultiRotorHopOnMDP(dynamics,multi_rotor_hopon_actions)
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

    # First check if control has been transfered - true
    if s.control_transfer == true || s.horizon == 0
        return true
    end

    return false

end

function terminalreward(mdp::ControlledMultiRotorHopOnMDP, s::ControlledHopOnStateAugmented)

    if !isterminal(mdp,s)
        return 0.0
    end

    if s.control_transfer == true
        return -CONTROL_TRANSFER_PENALTY
    end

    curr_pos = Point(s.rel_uavstate.x, s.rel_uavstate.y)

    if point_norm(curr_pos) < HOP_DISTANCE_THRESHOLD
        return HOP_REWARD
    end

    # Could not reach the car in time to hop
    return -HOP_REWARD

end

# This has to be INDEPENDENT of whether car has reached end
function isterminal(mdp::ControlledMultiRotorHopOnMDP, s::ControlledHopOnState)

    if s.control_transfer == true
        return true
    end

    return false
end

function terminalreward(mdp::ControlledMultiRotorHopOnMDP, s::ControlledHopOnState)

    if !isterminal(mdp,s)
        return 0.0
    end

    if s.control_transfer == true
        return -CONTROL_TRANSFER_PENALTY
    end

    curr_pos = Point(s.rel_uavstate.x, s.rel_uavstate.y)

    if point_norm(curr_pos) < HOP_DISTANCE_THRESHOLD
        return HOP_REWARD
    end

    # Could not reach the car in time to hop
    return -HOP_REWARD
end


function generate_sr(mdp::ControlledMultiRotorHopOnMDP, s::ControlledHopOnState, a::HopOnAction, rng::RNG=Base.GLOBAL_RNG) where {RNG <: AbstractRNG}

    cost = TIME_COEFFICIENT*MDP_TIMESTEP

    # Depending on action, do various things
    if a.uavaction != nothing
        new_uavstate = next_state(mdp.dynamics, s.rel_uavstate, a.uavaction, rng)

        cost += dynamics_cost(mdp.dynamics, s.rel_uavstate, new_uavstate)

        if s.control_transfer == true
            throw(ErrorException("Can't have control true here"))
        end

        return ControlledHopOnState(new_uavstate, false), -cost
    elseif a.control_transfer == true
        # Transfer control to higher layer
        return ControlledHopOnState(s.rel_uavstate, true), -cost
    else
        throw(ArgumentError("Invalid action specified!"))
    end
end


### TRANSITION ###
# Monte Carlo, will also need to do sigma point sampling later
# NOTE - This is for training, for actual will append horizon
# to create multiple augmented states.
function generate_sr(mdp::ControlledMultiRotorHopOnMDP, s::ControlledHopOnStateAugmented, a::HopOnAction, rng::RNG=Base.GLOBAL_RNG) where {RNG <: AbstractRNG}

    cost = TIME_COEFFICIENT*MDP_TIMESTEP

    # Can assume control_transfer is false

    # Depending on action, do various things
    if a.uavaction != nothing
        new_uavstate = next_state(mdp.dynamics, s.rel_uavstate, a.uavaction, rng)
        cost += dynamics_cost(mdp.dynamics, s.rel_uavstate, new_uavstate)

        if s.control_transfer == true
            throw(ErrorException("Can't have control true here"))
        end

        return ControlledHopOnStateAugmented(new_uavstate, false, s.horizon-1), -cost
    elseif a.control_transfer == true
        # Transfer control to higher layer
        return ControlledHopOnStateAugmented(s.rel_uavstate, true, s.horizon-1), -cost
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
            nbr_states[i] = ControlledHopOnStateAugmented(sigma_uavstates[i], s.control_transfer, s.horizon-1)
        end

        return SparseCat(nbr_states, sigma_probs)
    elseif a.control_transfer == true
        return SparseCat([ControlledHopOnStateAugmented(s.rel_uavstate, true, s.horizon-1)],[1.0])
    else
        throw(ArgumentError("Invalid action specified!"))
    end
end

function reward(mdp::ControlledMultiRotorHopOnMDP, s::ControlledHopOnStateAugmented, a::HopOnAction, sp::ControlledHopOnStateAugmented)

    cost = TIME_COEFFICIENT*MDP_TIMESTEP

    if a.uavaction != nothing

        old_point::Point = Point(s.rel_uavstate.x, s.rel_uavstate.y)
        new_point::Point = Point(sp.rel_uavstate.x, sp.rel_uavstate.y)
        dyn_dist::Float64 = point_dist(old_point, new_point)

        if dyn_dist < EPSILON && sqrt(s.rel_uavstate.xdot^2 + s.rel_uavstate.ydot^2) < EPSILON
            cost += HOVER_COEFFICIENT*MDP_TIMESTEP
        else
            cost += FLIGHT_COEFFICIENT*dyn_dist
        end
    end

    # For the other action, no additional cost accrued
    return -cost
end

function generate_start_state(mdp::ControlledMultiRotorHopOnMDP, rng::RNG=Base.GLOBAL_RNG) where {RNG <: AbstractRNG}

    uav_startstate = generate_start_state(mdp.dynamics)

    return ControlledHopOnState(uav_startstate,false)
end

# State conversion functions
function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, s::ControlledHopOnStateAugmented, mdp::ControlledMultiRotorHopOnMDP)
  v = SVector{6,Float64}(s.rel_uavstate.x, s.rel_uavstate.y, s.rel_uavstate.xdot, s.rel_uavstate.ydot,
                         convert(Float64,s.control_transfer), convert(Float64,s.horizon))
  return v
end

function POMDPs.convert_s(::Type{ControlledHopOnStateAugmented}, v::AbstractVector{Float64}, mdp::ControlledMultiRotorHopOnMDP)
  s = ControlledHopOnStateAugmented(MultiRotorUAVState(v[1],v[2],v[3],v[4]),convert(Bool,v[5]), convert(Int64,v[6]))
  return s
end