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


# Compute worst dynamics reward to goal
single_step_worst_reward = Inf
dynamics_actions = Vector{MultiRotorUAVAction}()
acc_vals = linspace(-ACCELERATION_LIM,ACCELERATION_LIM,ACCELERATION_NUMVALS)

info("Creating dynamics actions vector")
for xddot in acc_vals
    for yddot in acc_vals
        push!(dynamics_actions,MultiRotorUAVAction(xddot,yddot))
    end
end

info("Iterating over all S-A pairs")
for action in dynamics_actions
    for (i,vect) in enumerate(nohor_gridverts)
        state = convert_s(MultiRotorUAVState, vect, pc_hopon_mdp)
        this_reward = 0.0

        for _ = 1:MC_GENERATIVE_NUMSAMPLES
            sp = next_state(uav_dynamics, state, action, rng)
            this_reward += -(1.0 - energy_time_alpha)*dynamics_cost(uav_dynamics, state, sp)
        end
        this_reward = this_reward / MC_GENERATIVE_NUMSAMPLES - energy_time_alpha*TIME_COEFFICIENT*MDP_TIMESTEP

        if this_reward < single_step_worst_reward
            single_step_worst_reward = this_reward
        end
    end
end

worst_dynamics_reward = single_step_worst_reward*HORIZON_LIM

no_hop_penalty = -worst_dynamics_reward
# println("No hop penalty - ",no_hop_penalty)

pc_hopon_mdp.no_hop_penalty = no_hop_penalty


# Run LocalApproxVI to get VFA
# For out-horizon, don't need to rerun - there is no out-horizon abort action because it should not abort
# when it is out of horizon
# Create the full in-horizon and out-horizon grid
hopon_grid = RectangleGrid(xy_spacing, xy_spacing, xydot_spacing, xydot_spacing,0 : 1 : HORIZON_LIM)
grid_vertices = vertices(hopon_grid)
println(length(grid_vertices)," hopon vertices!")



info("Solving in-horizon policy")
in_hor_approx = LocalGIFunctionApproximator(hopon_grid)
approx_hopon_inhorizon_solver = LocalApproximationValueIterationSolver(in_hor_approx, max_iterations=1, verbose=true,
                                                                       rng=rng,is_mdp_generative=true, n_generative_samples=MC_GENERATIVE_NUMSAMPLES)
approx_hopon_inhorizon_policy = solve(approx_hopon_inhorizon_solver, pc_hopon_mdp)

##################### Do work specific for out of horizon policy
out_hor_approx_augmented = LocalGIFunctionApproximator(hopon_grid)
approx_hopon_outhorizon_solver_augmented = LocalApproximationValueIterationSolver(out_hor_approx_augmented, max_iterations=1, verbose=true,rng=rng,
                                                                        is_mdp_generative=true, n_generative_samples=MC_GENERATIVE_NUMSAMPLES)

# NOTE : Set terminal costs to off
pc_hopon_mdp_outhor = ControlledMultiRotorHopOnMDP(uav_dynamics)
pc_hopon_mdp_outhor.terminal_costs_set = false
approx_hopon_outhorizon_policy_augmented = solve(approx_hopon_outhorizon_solver_augmented, pc_hopon_mdp_outhor)

hopon_grid_outhor = RectangleGrid(xy_spacing, xy_spacing, xydot_spacing, xydot_spacing, [HORIZON_LIM+1])
all_interp_values = get_all_interpolating_values(approx_hopon_outhorizon_policy_augmented.interp)
all_interp_states = get_all_interpolating_points(approx_hopon_outhorizon_policy_augmented.interp)


outhor_interp_values = Vector{Float64}()
# Copy over the values for horizon = k
for (v,s) in zip(all_interp_values,all_interp_states)
    if s[end] == HORIZON_LIM
        push!(outhor_interp_values,v)
    end
end

@assert length(outhor_interp_values) == length(hopon_grid_outhor)
out_hor_approx_true = LocalGIFunctionApproximator(hopon_grid_outhor,outhor_interp_values)
approx_hopon_outhorizon_policy = LocalApproximationValueIterationPolicy(out_hor_approx_true,ordered_actions(pc_hopon_mdp_outhor),
                                                            pc_hopon_mdp_outhor, approx_hopon_outhorizon_solver_augmented.is_mdp_generative,
                                                            approx_hopon_outhorizon_solver_augmented.n_generative_samples,rng)

hopon_policy_preabort = PartialControlHopOnOffPolicy(approx_hopon_inhorizon_policy, approx_hopon_outhorizon_policy, ordered_actions(pc_hopon_mdp))

preabort_filename = string(policy_name,"-",poly_or_exp,"-preabort.jld")
save(preabort_filename,"hopon_policy",hopon_policy_preabort)


# hopon_policy_preabort = load("hopon_generative_preabort_unitgrid_paramset_trial.jld","policy")
# approx_hopon_inhorizon_policy = hopon_policy_preabort.in_horizon_policy
# approx_hopon_outhorizon_policy = hopon_policy_preabort.out_horizon_policy
# hopon_grid = RectangleGrid(xy_spacing, xy_spacing,
#             xydot_spacing, xydot_spacing,0 : 1 : HORIZON_LIM)


# Now iterate over each horizon value (1 to LIM), choose worst value at each point (you know it is negative)
# and set horizon abort penalty to be greater than BETA*worst(h)
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


info("Re-solving in-horizon policy with abort penalty")
# Need to reinit grid AND set terminal costs to true 
in_hor_approx_reinit = LocalGIFunctionApproximator(hopon_grid)
pc_hopon_mdp.terminal_costs_set = true
approx_hopon_inhorizon_solver_reinit = LocalApproximationValueIterationSolver(in_hor_approx_reinit, max_iterations=1, verbose=true,rng=rng,
                                                                       is_mdp_generative=true, n_generative_samples=MC_GENERATIVE_NUMSAMPLES)
approx_hopon_inhorizon_policy_reinit = solve(approx_hopon_inhorizon_solver_reinit, pc_hopon_mdp)

policy_filename = string(policy_name,"-",poly_or_exp,"-abort_thresh-",abort_risk_threshold,".jld")
# # Now create full policy
hopon_policy = PartialControlHopOnOffPolicy(approx_hopon_inhorizon_policy_reinit, approx_hopon_outhorizon_policy, ordered_actions(pc_hopon_mdp))
save(policy_filename,"hopon_policy",hopon_policy)
