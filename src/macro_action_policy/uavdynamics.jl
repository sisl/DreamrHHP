abstract type UAVDynamicsModel end
abstract type UAVState end
abstract type UAVAction end

mutable struct MultiRotorUAVState <: UAVState
    x::Float64
    y::Float64
    xdot::Float64
    ydot::Float64
end

mutable struct MultiRotorUAVAction <: UAVAction
    xddot::Float64
    yddot::Float64
end

# Actions are defined as accelerations OR hop-on/hop-off
@enum HOP_ACTION HOPON=1 STAY=0 HOPOFF=-1

mutable struct MultiRotorUAVDynamicsModel <: UAVDynamicsModel
    timestep::Float64
    noise::Distributions.Normal{Float64}
end

function MultiRotorUAVDynamicsModel(t::Float64, sig::Float64)
    return MultiRotorUAVDynamicsModel(t,Distributions.Normal(0.0,sig))
end

function generate_start_state(model::MultiRotorUAVDynamicsModel, rng::RNG=Base.GLOBAL_RNG) where {RNG <: AbstractRNG}
    x = rand(rng,Uniform(-XY_LIM,XY_LIM))
    y = rand(rng,Uniform(-XY_LIM,XY_LIM))
    xdot = 0.
    ydot = 0.

    return MultiRotorUAVState(x,y,xdot,ydot)
end

# For some known start position
function get_state_at_rest(model::MultiRotorUAVDynamicsModel, p::Point)
    return MultiRotorUAVState(p.x, p.y, 0.0, 0.0)
end

function get_relative_state_to_goal(model::MultiRotorUAVDynamicsModel, goal_pos::Point, state::MultiRotorUAVState)
    return MultiRotorUAVState(state.x - goal_pos.x, state.y - goal_pos.y, state.xdot, state.ydot)
end



# Truncate positions to their bounds
# function truncated_state(state::MultiRotorUAVState)

#     if state.xdot < -XYDOT_LIM
#         txdot = -XYDOT_LIM
#     elseif state.xdot > XYDOT_LIM
#         txdot = XYDOT_LIM
#     else
#         txdot = state.xdot
#     end

#     if state.ydot < -XYDOT_LIM
#         tydot = -XYDOT_LIM
#     elseif state.ydot > XYDOT_LIM
#         tydot = XYDOT_LIM
#     else
#         tydot = state.ydot
#     end

#     return MultiRotorUAVState(state.x,state.y,txdot,tydot)

# end

function truncate_vel(vel::Float64)

    if vel < -XYDOT_LIM
        return -XYDOT_LIM
    elseif vel > XYDOT_LIM
        return XYDOT_LIM
    end
    
    return vel
end

function apply_controls(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, xddot::Float64, yddot::Float64)

    # Update position and velocity exactly
    xdot = truncate_vel(state.xdot + xddot*model.timestep)
    ydot = truncate_vel(state.ydot + yddot*model.timestep)

    # Get true effective accelerations
    true_xddot = (xdot - state.xdot)/model.timestep
    true_yddot = (ydot - state.ydot)/model.timestep

    x = state.x + state.xdot*model.timestep + 0.5*true_xddot*(model.timestep^2)
    y = state.y + state.ydot*model.timestep + 0.5*true_yddot*(model.timestep^2)

    return MultiRotorUAVState(x,y,xdot,ydot)
end


function next_state(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, action::MultiRotorUAVAction, rng::RNG=Base.GLOBAL_RNG) where {RNG <: AbstractRNG}
    
    # Sample noisy acceleration
    xddot = action.xddot + rand(rng,model.noise)
    yddot = action.yddot + rand(rng,model.noise)

    return apply_controls(model, state, xddot, yddot)
end


function dynamics_cost(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, next_state::MultiRotorUAVState)
    old_point::Point = Point(state.x, state.y)
    new_point::Point = Point(next_state.x, next_state.y)
    dyn_dist = point_dist(old_point, new_point)

    cost = 0.0

    if dyn_dist < model.timestep*EPSILON && sqrt(next_state.xdot^2 + next_state.ydot^2) < model.timestep*EPSILON
        cost += HOVER_COEFFICIENT*model.timestep
    else
        cost += FLIGHT_COEFFICIENT*dyn_dist
    end

    return cost
end

function sigma_point_states_weights(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, action::MultiRotorUAVAction)

    n = 2 # dimensionality
    lambd = SIGMA_PT_ALPHA^2*(n + SIGMA_PT_KAPPA) - n
    covar_mat = std(model.noise)^2 * eye(n)
    sigma_pt_mat = sqrt.((n+lambd)*covar_mat) 
    mean_row = [mean(model.noise), mean(model.noise)]

    sigma_xy_ddots = zeros((2*n+1,2))
    weights = zeros(2*n+1)

    sigma_xy_ddots[1,:] = [action.xddot, action.yddot]
    weights[1] = lambd/(n+lambd)

    for i=2:n+1
        sigma_xy_ddots[i,:] = mean_row + sigma_pt_mat[i-1,:]
        weights[i] = 1.0/(2*(n+lambd))
    end

    for i=n+2:2*n+1
        sigma_xy_ddots[i,:] = mean_row - sigma_pt_mat[i-n-1,:]
        weights[i] = 1.0/(2*(n+lambd))
    end

    # Now apply accelerations to get the states
    sigma_states = Vector{MultiRotorUAVState}(2*n+1)
    for i = 1:2*n+1
        sigma_states[i] = apply_controls(model, state, sigma_xy_ddots[i,1], sigma_xy_ddots[i,2])
    end

    return sigma_states, weights
end


function get_hopon_pos_state(position::Point)
    return MultiRotorUAVState(position.x, position.y, 0., 0.)
end