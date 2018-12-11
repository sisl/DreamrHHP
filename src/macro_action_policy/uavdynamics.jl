abstract type UAVDynamicsModel end
abstract type UAVState end
abstract type UAVAction end

"""
Represents state of simple 2D multirotor, with position and velocity in each direction.
"""
struct MultiRotorUAVState <: UAVState
    x::Float64
    y::Float64
    xdot::Float64
    ydot::Float64
end

function get_position(state::MultiRotorUAVState)
    return Point(state.x, state.y)
end

function get_speed(state::MultiRotorUAVState)
    return sqrt(state.xdot^2 + state.ydot^2)
end 

"""
Represents control action for simple 2D multirotor, with acceleration in each direction.
"""
struct MultiRotorUAVAction <: UAVAction
    xddot::Float64
    yddot::Float64
end

"""
Actions are defined as accelerations OR stay/hop-on/hop-off
"""
@enum HOP_ACTION HOPON=1 STAY=0 HOPOFF=-1


"""
Simulates physical dynamics of UAV, mapping state and control to next state
"""
struct MultiRotorUAVDynamicsModel <: UAVDynamicsModel
    timestep::Float64
    noise::Distributions.ZeroMeanDiagNormal
    params::Parameters
end

"""
    MultiRotorUAVDynamicsModel(t::Float64, params::Parameters, sig::Float64)

Define a dynamics model where t is the timestep duration, params is the collection of system parameters,
and sig is the standard deviation along each axis (sig_xy) is vector of standard deviations for each corresponding axis
"""
function MultiRotorUAVDynamicsModel(t::Float64, sig::Float64, params::Parameters)
    # Generate diagonal covariance matrix
    noise = Distributions.MvNormal([sig,sig])
    return MultiRotorUAVDynamicsModel(t, noise, params)
end

function MultiRotorUAVDynamicsModel(t::Float64, sig_xy::StaticVector{2,Float64}, params::Parameters)
    # Generate diagonal covariance matrix
    noise = Distributions.MvNormal([sig[1],sig[2]])
    return MultiRotorUAVDynamicsModel(t, noise, params)
end

"""
    get_uav_dynamics_actions(model::MultiRotorUAVDynamicsModel)

Compiles a vector of all multirotor acceleration actions within limits, based on the resolution parameters
"""
function get_uav_dynamics_actions(model::MultiRotorUAVDynamicsModel)

    multirotor_actions = Vector{MultiRotorUAVAction}(undef,0)
    acc_vals = range(-model.params.scale_params.ACCELERATION_LIM,stop=model.params.scale_params.ACCELERATION_LIM,length=model.params.scale_params.ACCELERATION_NUMVALS)

    for xddot in acc_vals
        for yddot in acc_vals
            push!(multirotor_actions,MultiRotorUAVAction(xddot, yddot))
        end
    end

    return multirotor_actions
end

"""
    generate_start_state(model::MultiRotorUAVDynamicsModel, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

Generate a MultiRotorUAVState with a random location inside the grid and at rest
"""
function generate_start_state(model::MultiRotorUAVDynamicsModel, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    x = rand(rng,Uniform(-model.params.scale_params.XY_LIM,model.params.scale_params.XY_LIM))
    y = rand(rng,Uniform(-model.params.scale_params.XY_LIM,model.params.scale_params.XY_LIM))
    xdot = 0.
    ydot = 0.

    return MultiRotorUAVState(x,y,xdot,ydot)
end

"""
    get_state_at_rest(model::MultiRotorUAVDynamicsModel, p::Point)

Given a Point instance, generate the corresponding MultiRotorUAVState at rest at that position
"""
function get_state_at_rest(model::MultiRotorUAVDynamicsModel, p::Point)
    return MultiRotorUAVState(p.x, p.y, 0.0, 0.0)
end

"""
get_relative_state_to_goal(model::MultiRotorUAVDynamicsModel, goal_pos::Point, state::MultiRotorUAVState)

Given some goal position, get the MultiRotorUAVState with relative position and own velocity
"""
function get_relative_state_to_goal(model::MultiRotorUAVDynamicsModel, goal_pos::Point, state::MultiRotorUAVState)
    return MultiRotorUAVState(state.x - goal_pos.x, state.y - goal_pos.y, state.xdot, state.ydot)
end

"""
    apply_controls(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, xddot::Float64, yddot::Float64)

Propagate the MultiRotorUAVState through the dynamics model (without noise)
"""
function apply_controls(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, xddot::Float64, yddot::Float64)

    # Update position and velocity exactly
    xdot = truncate_vel(state.xdot + xddot*model.timestep, model.params.scale_params.XYDOT_LIM)
    ydot = truncate_vel(state.ydot + yddot*model.timestep, model.params.scale_params.XYDOT_LIM)

    # Get true effective accelerations
    true_xddot = (xdot - state.xdot)/model.timestep
    true_yddot = (ydot - state.ydot)/model.timestep

    x = state.x + state.xdot*model.timestep + 0.5*true_xddot*(model.timestep^2)
    y = state.y + state.ydot*model.timestep + 0.5*true_yddot*(model.timestep^2)

    return MultiRotorUAVState(x,y,xdot,ydot)
end

"""
    next_state(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, action::MultiRotorUAVAction, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

Generate the next state by propagating the action noisily with the dynamics
"""
function next_state(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, action::MultiRotorUAVAction, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}

    # Sample acceleration noise vector    
    noiseval = rand(rng,model.noise)

    # Augment actual acceleration with noise
    xddot = action.xddot + noiseval[1]
    yddot = action.yddot + noiseval[2]

    return apply_controls(model, state, xddot, yddot)
end


"""
    function dynamics_cost(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, next_state::MultiRotorUAVState)

Compute the energy/dynamics cost of going from one state to another, based on the cost parameters in the model.
N.B - In the most general case, cost should depend on the action too.
"""
function dynamics_cost(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, next_state::MultiRotorUAVState)
    old_point = Point(state.x, state.y)
    new_point = Point(next_state.x, next_state.y)
    dyn_dist = point_dist(old_point, new_point)

    cost = 0.0

    # Compute cost due to flying or due to hovering
    if dyn_dist < model.timestep*model.params.scale_params.EPSILON &&
        sqrt(next_state.xdot^2 + next_state.ydot^2) < model.timestep*model.params.scale_params.EPSILON
        cost += model.params.cost_params.HOVER_COEFFICIENT*model.timestep
    else
        cost += model.params.cost_params.FLIGHT_COEFFICIENT*dyn_dist
    end

    return cost
end

# DEPRECATED
# function sigma_point_states_weights(model::MultiRotorUAVDynamicsModel, state::MultiRotorUAVState, action::MultiRotorUAVAction)

#     n = 2 # dimensionality
#     lambd = SIGMA_PT_ALPHA^2*(n + SIGMA_PT_KAPPA) - n
#     covar_mat = std(model.noise)^2 * eye(n)
#     sigma_pt_mat = sqrt.((n+lambd)*covar_mat) 
#     mean_row = [mean(model.noise), mean(model.noise)]

#     sigma_xy_ddots = zeros((2*n+1,2))
#     weights = zeros(2*n+1)

#     sigma_xy_ddots[1,:] = [action.xddot, action.yddot]
#     weights[1] = lambd/(n+lambd)

#     for i=2:n+1
#         sigma_xy_ddots[i,:] = mean_row + sigma_pt_mat[i-1,:]
#         weights[i] = 1.0/(2*(n+lambd))
#     end

#     for i=n+2:2*n+1
#         sigma_xy_ddots[i,:] = mean_row - sigma_pt_mat[i-n-1,:]
#         weights[i] = 1.0/(2*(n+lambd))
#     end

#     # Now apply accelerations to get the states
#     sigma_states = Vector{MultiRotorUAVState}(2*n+1)
#     for i = 1:2*n+1
#         sigma_states[i] = apply_controls(model, state, sigma_xy_ddots[i,1], sigma_xy_ddots[i,2])
#     end

#     return sigma_states, weights
# end