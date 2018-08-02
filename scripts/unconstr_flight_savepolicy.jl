using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration
using POMDPs
using POMDPModels
using POMDPToolbox
using StaticArrays
using JLD

using HitchhikingDrones

# TODO - EDIT
rng = MersenneTwister(5)
DISCOUNT = 1.0
policy_name = ARGS[1]
poly_or_exp = ARGS[2]

policy_name = string(policy_name,"-",poly_or_exp)

# Create MDP - Need Dynamics Model first
uav_dynamics = MultiRotorUAVDynamicsModel(MDP_TIMESTEP, ACC_NOISE_STD)
flight_mdp = UnconstrainedFlightMDP(uav_dynamics, DISCOUNT)

if poly_or_exp == "poly"
    xy_spacing = polyspace_symmetric(XY_LIM, XY_AXISVALS)
    xydot_spacing = polyspace_symmetric(XYDOT_LIM, XYDOT_AXISVALS)
elseif poly_or_exp == "exp"
    xy_spacing = log2space_symmetric(XY_LIM, XY_AXISVALS)
    xydot_spacing = log2space_symmetric(XYDOT_LIM, XYDOT_AXISVALS)
end

flight_grid = RectangleGrid(xy_spacing, xy_spacing, xydot_spacing, xydot_spacing)
grid_vertices = vertices(flight_grid)
println(length(grid_vertices)," vertices!")


flight_approximator = LocalGIFunctionApproximator(flight_grid)
approx_flight_solver = LocalApproximationValueIterationSolver(flight_approximator,max_iterations=25,verbose=true,rng=rng,
                                                is_mdp_generative=true,n_generative_samples=MC_GENERATIVE_NUMSAMPLES)
approx_flight_policy = solve(approx_flight_solver, flight_mdp)
policy_filename = string(policy_name,".jld")
save(policy_filename, "flight_policy", approx_flight_policy)
