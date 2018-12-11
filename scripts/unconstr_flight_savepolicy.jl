using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration
using POMDPs
using POMDPModels
using POMDPModelTools
using POMDPPolicies
using StaticArrays
using JLD2, FileIO
using Random
using Distributions
using PDMats
using HitchhikingDrones

# Comment if testing in REPL
# ARGS = String["../data/paramsets/scale-small-test.toml","../data/paramsets/simtime-small-test.toml",
#                 "../data/paramsets/cost-1.toml","../data/policies/test-uf","poly","0.75"]


rng = MersenneTwister(5)
DISCOUNT = 1.0

# Call as julia <script.jl> <scalefile,timefile,costfile> <policy-prefix> <kind-of-approx> <alpha> 
scale_file = ARGS[1]
simtime_file = ARGS[2]
cost_file = ARGS[3]
policy_prefix = ARGS[4]
poly_or_exp = ARGS[5]
energy_time_alpha = parse(Float64,ARGS[6])


# First, parse parameter files to get filenames and construct params object
params = parse_params(scale_file=scale_file, simtime_file=simtime_file, cost_file=cost_file)

# Next, construct name of policy
policy_name = string(policy_prefix,"-",poly_or_exp,"-alpha-",energy_time_alpha)
@show policy_name

# Create MDP - Need Dynamics Model first
uav_dynamics = MultiRotorUAVDynamicsModel(params.time_params.MDP_TIMESTEP, params.scale_params.ACC_NOISE_STD, params)
flight_mdp = UnconstrainedFlightMDP{MultiRotorUAVState,MultiRotorUAVAction}(uav_dynamics, DISCOUNT, energy_time_alpha)

if poly_or_exp == "poly"
    xy_spacing = polyspace_symmetric(params.scale_params.XY_LIM, params.scale_params.XY_AXISVALS)
    xydot_spacing = polyspace_symmetric(params.scale_params.XYDOT_LIM, params.scale_params.XYDOT_AXISVALS)
elseif poly_or_exp == "exp"
    xy_spacing = log2space_symmetric(params.scale_params.XY_LIM, params.scale_params.XY_AXISVALS)
    xydot_spacing = log2space_symmetric(params.scale_params.XYDOT_LIM, params.scale_params.XYDOT_AXISVALS)
end

# Create grid of vertices for local interpolation
flight_grid = RectangleGrid(xy_spacing, xy_spacing, xydot_spacing, xydot_spacing)
grid_vertices = vertices(flight_grid)

# Create function approximator and compute policy
flight_approximator = LocalGIFunctionApproximator(flight_grid)
approx_flight_solver = LocalApproximationValueIterationSolver(flight_approximator,max_iterations=2,verbose=true,rng=rng,
                                                is_mdp_generative=true,n_generative_samples=params.scale_params.MC_GENERATIVE_NUMSAMPLES)
approx_flight_policy = solve(approx_flight_solver, flight_mdp)

# Save policy to filename
policy_filename = string(policy_name,".jld2")
save_localapproxvi_policy_to_jld2(policy_filename, approx_flight_policy, flight_mdp, 5)