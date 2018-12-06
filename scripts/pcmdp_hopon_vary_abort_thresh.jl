using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration
using POMDPs
using POMDPModels
using POMDPToolbox
using StaticArrays
using JLD
using HitchhikingDrones

policy_name = ARGS[1]
poly_or_exp = ARGS[2]
abort_risk_threshold = parse(Float64,ARGS[3])
energy_time_alpha = parse(Float64,ARGS[4])

# For consistency
rng = MersenneTwister(5)

# Create MDP - Need Dynamics Model first
uav_dynamics = MultiRotorUAVDynamicsModel(MDP_TIMESTEP, ACC_NOISE_STD, HOVER_COEFFICIENT, FLIGHT_COEFFICIENT)
pc_hopon_mdp = ControlledMultiRotorHopOnMDP(uav_dynamics,energy_time_alpha)

if poly_or_exp == "poly"
    xy_spacing = polyspace_symmetric(XY_LIM, XY_AXISVALS)
    xydot_spacing = polyspace_symmetric(XYDOT_LIM, XYDOT_AXISVALS)
elseif poly_or_exp == "exp"
    xy_spacing = log2space_symmetric(XY_LIM, XY_AXISVALS)
    xydot_spacing = log2space_symmetric(XYDOT_LIM, XYDOT_AXISVALS)
end

# Create pseudo grid WITHOUT horizon axis
no_horizon_hopongrid = RectangleGrid(xy_spacing, xy_spacing, xydot_spacing, xydot_spacing)
nohor_gridverts = vertices(no_horizon_hopongrid)


# Load preabort policy
preabort_filename = string(policy_name,"-",poly_or_exp,"-preabort.jld")
hopon_policy_preabort = load(preabort_filename,"hopon_policy")
approx_hopon_inhorizon_policy = hopon_policy_preabort.in_horizon_policy
approx_hopon_outhorizon_policy = hopon_policy_preabort.out_horizon_policy
hopon_grid = RectangleGrid(xy_spacing, xy_spacing,
            xydot_spacing, xydot_spacing,0 : 1 : HORIZON_LIM)

println(preabort_filename)

# Now do the second half of full PCMDP stuff, with abort threshold
for h = 1 : HORIZON_LIM
    worst_val = 0.0
    best_val = -Inf
    for (i,vect) in enumerate(nohor_gridverts)
        hor_vect = Vector{Float64}(vect)
        push!(hor_vect, h)

        # Lookup value from in_hor_approx
        state_val = compute_value(approx_hopon_inhorizon_policy.interp, hor_vect)

        worst_val = (state_val < worst_val)? state_val : worst_val
        best_val = (state_val > best_val)? state_val : best_val
    end

    println("Horizon - ",h," : worst val - ",worst_val," best val - ",best_val)

    # Now set penalty to be BETA * worst-val
    pc_hopon_mdp.horizon_abort_penalty[h] = abort_risk_threshold*abs(worst_val)
end

println("Re-solving in-horizon policy with abort penalty")
# Need to reinit grid AND set terminal costs to true 
in_hor_approx_reinit = LocalGIFunctionApproximator(hopon_grid)
pc_hopon_mdp.terminal_costs_set = true
approx_hopon_inhorizon_solver_reinit = LocalApproximationValueIterationSolver(in_hor_approx_reinit, max_iterations=1, verbose=false,rng=rng,
                                                                       is_mdp_generative=true, n_generative_samples=MC_GENERATIVE_NUMSAMPLES)
approx_hopon_inhorizon_policy_reinit = solve(approx_hopon_inhorizon_solver_reinit, pc_hopon_mdp)

policy_filename = string(policy_name,"-",poly_or_exp,"-abort_thresh-",abort_risk_threshold,".jld")
# # Now create full policy
hopon_policy = PartialControlHopOnOffPolicy(approx_hopon_inhorizon_policy_reinit, approx_hopon_outhorizon_policy, ordered_actions(pc_hopon_mdp))
save(policy_filename,"hopon_policy",hopon_policy)