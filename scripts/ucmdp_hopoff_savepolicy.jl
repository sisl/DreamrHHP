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
uc_hopoff_mdp = HopOffMDP()
rng = MersenneTwister(5)
hopoff_grid_inhor = RectangleGrid([0, 1], 0:1:HORIZON_LIM)
in_hor_grid_verts = vertices(hopoff_grid_inhor)

in_hor_approx = LocalGIFunctionApproximator(hopoff_grid_inhor)
approx_hopoff_inhorizon_solver = LocalApproximationValueIterationSolver(in_hor_approx, max_iterations=1, verbose=true,
                                                           rng=rng, is_mdp_generative=false, n_generative_samples=0)
approx_hopoff_inhorizon_policy = solve(approx_hopoff_inhorizon_solver, uc_hopoff_mdp)


uc_hopoff_mdp_outhor = HopOffMDP()
uc_hopoff_mdp_outhor.terminal_costs_set = false
out_hor_approx_augmented = LocalGIFunctionApproximator(hopoff_grid_inhor)
approx_hopoff_outhorizon_solver_augmented = LocalApproximationValueIterationSolver(out_hor_approx_augmented, max_iterations=1, verbose=true,rng=rng,
                                                            is_mdp_generative=false, n_generative_samples=0)
solve(approx_hopoff_outhorizon_solver_augmented, uc_hopoff_mdp_outhor)

hopoff_grid_outhor = hopoff_grid_inhor = RectangleGrid([0, 1], [HORIZON_LIM+1])
all_interp_values = get_all_interpolating_values(approx_hopoff_outhorizon_solver_augmented.interp)
all_interp_states = get_all_interpolating_points(approx_hopoff_outhorizon_solver_augmented.interp)
outhor_interp_values = Vector{Float64}()

for (v,s) in zip(all_interp_values,all_interp_states)
  if s[end] == HORIZON_LIM
    push!(outhor_interp_values,v)
  end
end

@assert length(outhor_interp_values) == length(hopoff_grid_outhor)
out_hor_approx_true = LocalGIFunctionApproximator(hopoff_grid_outhor,outhor_interp_values)
approx_hopoff_outhorizon_policy = LocalApproximationValueIterationPolicy(out_hor_approx_true,ordered_actions(uc_hopoff_mdp_outhor),
                                        uc_hopoff_mdp_outhor, approx_hopoff_outhorizon_solver_augmented.is_mdp_generative,
                                        approx_hopoff_outhorizon_solver_augmented.n_generative_samples,rng)
hopoff_policy = PartialControlHopOnOffPolicy(approx_hopoff_inhorizon_policy, approx_hopoff_outhorizon_policy, ordered_actions(uc_hopoff_mdp))

policy_filename = string(policy_name,".jld")
save(policy_filename,"hopoff_policy",hopoff_policy)
