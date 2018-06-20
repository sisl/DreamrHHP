using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration
using POMDPs
using POMDPModels
using POMDPToolbox
using StaticArrays
using JLD

using HitchhikingDrones

DISCOUNT = 0.9

# Create MDP - Need Dynamics Model first
uav_dynamics = MultiRotorUAVDynamicsModel(MDP_TIMESTEP, ACC_NOISE_STD)
flight_mdp = UnconstrainedFlightMDP(uav_dynamics, DISCOUNT)

flight_grid = RectangleGrid(linspace(-XY_LIM,XY_LIM,XY_AXISVALS), linspace(-XY_LIM,XY_LIM,XY_AXISVALS), 
                           linspace(-XYDOT_LIM,XYDOT_LIM,XYDOT_AXISVALS), linspace(-XYDOT_LIM,XYDOT_LIM,XYDOT_AXISVALS))
grid_vertices = vertices(flight_grid)
println(length(grid_vertices)," vertices!")
grid_term_values = zeros(length(grid_vertices))
for (i,vect) in enumerate(grid_vertices)
    state = convert_s(MultiRotorUAVState, vect, flight_mdp)
    grid_term_values[i] = terminalreward(flight_mdp,state)
end

flight_approximator = LocalGIFunctionApproximator(flight_grid, grid_term_values)

approx_flight_solver = LocalApproximationValueIterationSolver(flight_approximator,max_iterations=30,verbose=true,
                                                        is_mdp_generative=true,n_generative_samples=MC_GENERATIVE_NUMSAMPLES,
                                                            terminal_costs_set=true)
approx_flight_policy = solve(approx_flight_solver, flight_mdp)

save("unconstr_flight_generative_unitgrid_paramset3.jld", "flight_policy", approx_flight_policy)