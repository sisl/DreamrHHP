"""
    truncate_vel(vel::Float64, xydot_lim::Float64)

Truncate velocity to [-lim, +lim] 
"""
function truncate_vel(vel::Float64, xydot_lim::Float64)

    if vel < -xydot_lim
        return -xydot_lim
    elseif vel > xydot_lim
        return xydot_lim
    end
    
    return vel
end

"""
    log2space_symmetric(symm_val::Float64, n::Int64, base_val::Int64=2)

Returns a vector of logarithmically spaced numbers from -x to +x, symmetrically
spaced around 0. The number of values must be odd to reflect 0 and a symmetric
number of arguments around it.
"""
function log2space_symmetric(symm_val::Float64, n::Int64, base_val::Int64=2)

    # Ensure that the number of terms is odd (so around 0.0)
    @assert n%2==1

    # Ensure that the value itself is positive
    @assert symm_val > 0

    vals = Vector{Float64}(undef,0)
    idx = 1
    midpt = convert(Int64,round((n-1)/2))

    for i=1:midpt
        push!(vals, -(symm_val/idx))
        idx = idx*base_val
    end

    # Add 0.0 to axis
    symm_vect = reverse(-vals)
    push!(vals, 0.0)
    append!(vals, symm_vect)

    return vals
end


"""
    polyspace_symmetric(symm_val::Float64, n::Int64, exp_val::Int64=3)

Returns a vector of polynomially spaced numbers from -x to +x, symmetrically
spaced around 0. The number of values must be odd to reflect 0 and a symmetric
number of arguments around it.
"""
function polyspace_symmetric(symm_val::Float64, n::Int64, exp_val::Int64=3)

    # Ensure that the number of terms is odd (so around 0.0)
    @assert n%2==1
    # Ensure that the value itself is positive
    @assert symm_val > 0

    vals = Vector{Float64}(undef,0)
    idx = 1
    midpt = convert(Int64,round((n-1)/2))

    x = (symm_val/(midpt^exp_val))^(1/exp_val)

    for i=midpt:-1:1
        val = -1*(i*x)^exp_val
        push!(vals,val)
    end

    symm_vect = reverse(-vals)
    push!(vals, 0.0)
    append!(vals, symm_vect)

    return vals
end

"""
Util for computing trivial straight line distance and time to reach goal
"""
function st_line_reward_time(dist::Float64)

    # For speedup and slow down - 2 * 1/2 * at^2
    acc_dist = ACCELERATION_LIM*(XYDOT_LIM/ACCELERATION_LIM)^2

    timeval = 2*(XYDOT_LIM/ACCELERATION_LIM) + (dist-acc_dist)/XYDOT_LIM

    reward = FLIGHT_COEFFICIENT*dist + TIME_COEFFICIENT*timeval
    
    return reward,timeval
end


## POLICY SAVING AND LOADING UTILS
function save_localapproxvi_policy_to_jld2(policy_fn::String, policy::LocalApproximationValueIterationPolicy, 
                                           mdp::Union{MDP,POMDP}, mersenne_seed::Int)

    save(policy_fn, "interp", policy.interp,
                    "mdp", mdp,
                    "is_mdp_generative",policy.is_mdp_generative,
                    "n_generative_samples",policy.n_generative_samples,
                    "mersenne_seed",mersenne_seed)
end

function load_localapproxvi_policy_from_jld2(policy_fn::String)

    policy_interp = load(policy_fn, "interp")
    policy_mdp = load(policy_fn, "mdp")
    policy_isgen = load(policy_fn,"is_mdp_generative")
    policy_n_gen_samples = load(policy_fn,"n_generative_samples")
    policy_seed = load(policy_fn, "mersenne_seed")

    return LocalApproximationValueIterationPolicy(policy_interp, ordered_actions(policy_mdp), policy_mdp, policy_isgen,
                                                  policy_n_gen_samples, MersenneTwister(policy_seed))
end