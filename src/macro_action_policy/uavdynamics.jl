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



# Truncate positions to their bounds
function truncated_state(state::MultiRotorUAVState)
    if state.x < -XY_LIM
        tx = -XY_LIM
    elseif state.x > XY_LIM
        tx = XY_LIM
    else
        tx = state.x
    end

    if state.y < -XY_LIM
        ty = -XY_LIM
    elseif state.y > XY_LIM
        ty = XY_LIM
    else
        ty = state.y
    end

    if state.xdot < -XYDOT_LIM
        txdot = -XYDOT_LIM
    elseif state.xdot > XYDOT_LIM
        txdot = XYDOT_LIM
    else
        txdot = state.xdot
    end

    if state.ydot < -XYDOT_LIM
        tydot = -XYDOT_LIM
    elseif state.ydot > XYDOT_LIM
        tydot = XYDOT_LIM
    else
        tydot = state.ydot
    end

    return MultiRotorUAVState(tx,ty,txdot,tydot)

end

function apply_controls(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, xddot::Float64, yddot::Float64)

    # Update position and velocity exactly
    x = state.x + state.xdot*model.timestep + 0.5*xddot*(model.timestep^2)
    y = state.y + state.ydot*model.timestep + 0.5*yddot*(model.timestep^2)
    xdot = state.xdot + xddot*model.timestep
    ydot = state.ydot + yddot*model.timestep

    return truncated_state(MultiRotorUAVState(x,y,xdot,ydot))
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

    if dyn_dist < EPSILON && sqrt(next_state.xdot^2 + next_state.ydot^2) < EPSILON
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