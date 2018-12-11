using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration
using POMDPs
using POMDPModels
using POMDPModelTools
using StaticArrays
using JLD2, FileIO
using Random
using Logging
using Distributions
using PDMats
using HitchhikingDrones


# For consistency
rng = MersenneTwister(5)

# Commented to test in REPL easily
# ARGS = String["../data/paramsets/scale-small-test.toml","../data/paramsets/simtime-small-test.toml",
#               "../data/paramsets/cost-1.toml","../data/policies/test-cf","poly","0.5","0.75"]

scale_file = ARGS[1]
simtime_file = ARGS[2]
cost_file = ARGS[3]
policy_name = ARGS[4]
poly_or_exp = ARGS[5]
abort_risk_threshold = parse(Float64,ARGS[6])
energy_time_alpha = parse(Float64,ARGS[7])

# First, parse parameter files to get filenames and construct params object
params = parse_params(scale_file=scale_file, simtime_file=simtime_file, cost_file=cost_file)

# Create MDP - Need Dynamics Model first
uav_dynamics = MultiRotorUAVDynamicsModel(params.time_params.MDP_TIMESTEP, params.scale_params.ACC_NOISE_STD, params)
pc_hopon_mdp = ControlledHopOnMDP{MultiRotorUAVState, MultiRotorUAVAction}(uav_dynamics, energy_time_alpha)

if poly_or_exp == "poly"
    xy_spacing = polyspace_symmetric(params.scale_params.XY_LIM, params.scale_params.XY_AXISVALS)
    xydot_spacing = polyspace_symmetric(params.scale_params.XYDOT_LIM, params.scale_params.XYDOT_AXISVALS)
elseif poly_or_exp == "exp"
    xy_spacing = log2space_symmetric(params.scale_params.XY_LIM, params.scale_params.XY_AXISVALS)
    xydot_spacing = log2space_symmetric(params.scale_params.XYDOT_LIM, params.scale_params.XYDOT_AXISVALS)
end

# Create pseudo grid WITHOUT horizon axis
no_horizon_hopongrid = RectangleGrid(xy_spacing, xy_spacing, xydot_spacing, xydot_spacing)
nohor_gridverts = vertices(no_horizon_hopongrid)


# Load preabort policy
@info "Loading pre-abort policies"
preabort_filename_inhor = string(policy_name,"-",poly_or_exp,"-preabort-inhor.jld2")
preabort_filename_outhor = string(policy_name,"-",poly_or_exp,"-preabort-outhor.jld2")

approx_hopon_inhorizon_policy = load_localapproxvi_policy_from_jld2(preabort_filename_inhor)
approx_hopon_outhorizon_policy = load_localapproxvi_policy_from_jld2(preabort_filename_outhor)

hopon_grid = RectangleGrid(xy_spacing, xy_spacing,
            xydot_spacing, xydot_spacing,0 : 1 : params.time_params.HORIZON_LIM)


# Now do the second half of full PCMDP stuff, with abort threshold
@info "Computing beta-dependent horizon abort penalties"
for h = 1 : params.time_params.HORIZON_LIM
    worst_val = 0.0
    best_val = -Inf
    for (i,vect) in enumerate(nohor_gridverts)
        hor_vect = Vector{Float64}(vect)
        push!(hor_vect, h)

        # Lookup value from in_hor_approx
        state_val = compute_value(approx_hopon_inhorizon_policy.interp, hor_vect)

        worst_val = (state_val < worst_val) ? state_val : worst_val
        best_val = (state_val > best_val) ? state_val : best_val
    end

    println("Horizon - ",h," : worst val - ",worst_val," best val - ",best_val)

    # Now set penalty to be BETA * worst-val
    pc_hopon_mdp.horizon_abort_penalty[h] = abort_risk_threshold*abs(worst_val)
end

@info "Re-solving in-horizon policy with abort penalty"
# Need to reinit grid AND set terminal costs to true 
in_hor_approx_reinit = LocalGIFunctionApproximator(hopon_grid)
pc_hopon_mdp.terminal_costs_set = true
approx_hopon_inhorizon_solver_reinit = LocalApproximationValueIterationSolver(in_hor_approx_reinit, max_iterations=1, verbose=false,rng=rng,
                                                                       is_mdp_generative=true, n_generative_samples=params.scale_params.MC_GENERATIVE_NUMSAMPLES)
approx_hopon_inhorizon_policy_reinit = solve(approx_hopon_inhorizon_solver_reinit, pc_hopon_mdp)

policy_filename = string(policy_name,"-",poly_or_exp,"-abort_thresh-",abort_risk_threshold,"-inhor.jld2")


## Now save new in-horizon policy (as you are not computing a new out-horizon policy)
save_localapproxvi_policy_to_jld2(policy_filename, approx_hopon_inhorizon_policy_reinit, pc_hopon_mdp, 5)

# # Now create full policy
# hopon_policy = PartialControlHopOnOffPolicy(approx_hopon_inhorizon_policy_reinit, approx_hopon_outhorizon_policy, ordered_actions(pc_hopon_mdp))
# save(policy_filename,"hopon_policy",hopon_policy)