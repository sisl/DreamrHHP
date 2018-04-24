using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration
using POMDPs
using POMDPModels
using POMDPToolbox
using StaticArrays
using JLD

using hitchhiking_drones

# State conversion functions
function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, s::ControlledHopOnStateAugmented, mdp::ControlledMultiRotorHopOnMDP)
  v = SVector{6,Float64}(s.uavstate.x, s.uavstate.y, s.uavstate.xdot, s.uavstate.ydot,
                         convert(Float64,s.control_transfer), convert(Float64,s.horizon))
  return v
end

function POMDPs.convert_s(::Type{ControlledHopOnStateAugmented}, v::AbstractVector{Float64}, mdp::ControlledMultiRotorHopOnMDP)
  s = ControlledHopOnStateAugmented(MultiRotorUAVState(v[1],v[2],v[3],v[4]),convert(Bool,v[5]), convert(Int64,v[6]))
  return s
end



# Create MDP - Need Dynamics Model first
uav_dynamics = MultiRotorUAVDynamicsModel(MDP_TIMESTEP, ACC_NOISE_STD)

pc_hopon_mdp = ControlledMultiRotorHopOnMDP(uav_dynamics)



# Create the grid - this will be used by both the in_horizon and out_of_horizon approximators
hopon_grid = RectangleGrid(-XY_LIM:XY_RES:XY_LIM, -XY_LIM:XY_RES:XY_LIM, -XYDOT_LIM:XYDOT_RES:XYDOT_LIM, -XYDOT_LIM:XYDOT_RES:XYDOT_LIM,
                            [0, 1], 0 : 1 : HORIZON_LIM)
grid_vertices = vertices(hopon_grid)


##################### Do work specific for in-horizon policy
in_hor_grid_term_values = zeros(length(grid_vertices))

for (i,vect) in enumerate(grid_vertices)
    state = convert_s(ControlledHopOnStateAugmented, vect, pc_hopon_mdp)
    in_hor_grid_term_values[i] = terminalreward(pc_hopon_mdp,state)
end

in_hor_approx = LocalGIFunctionApproximator(hopon_grid, in_hor_grid_term_values)

approx_hopon_inhorizon_solver = LocalApproximationValueIterationSolver(in_hor_approx, max_iterations=2, verbose=true,
                                                            is_mdp_generative=false, n_generative_samples=0,
                                                            terminal_costs_set=true)
approx_hopon_inhorizon_policy = solve(approx_hopon_inhorizon_solver, pc_hopon_mdp)

outhor_grid_termvals = zeros(length(grid_vertices))
for (i,vect) in enumerate(grid_vertices)
  state = convert_s(ControlledHopOnStateAugmented, vect, pc_hopon_mdp)
  if isterminal(pc_hopon_mdp, state)
    if state.control_transfer == true
      outhor_grid_termvals[i] = -CONTROL_TRANSFER_PENALTY
    end
  end
end


##################### Do work specific for out of horizon policy
out_hor_approx_augmented = LocalGIFunctionApproximator(hopon_grid, outhor_grid_termvals)
approx_hopon_outhorizon_solver_augmented = LocalApproximationValueIterationSolver(out_hor_approx_augmented, max_iterations=2, verbose=true,
                                                            is_mdp_generative=false, n_generative_samples=0,
                                                            terminal_costs_set=true)
solve(approx_hopon_outhorizon_solver_augmented, pc_hopon_mdp)

# While querying, just set horizon to 0 here
hopon_grid_outhor = RectangleGrid(-XY_LIM:XY_RES:XY_LIM, -XY_LIM:XY_RES:XY_LIM, -XYDOT_LIM:XYDOT_RES:XYDOT_LIM, -XYDOT_LIM:XYDOT_RES:XYDOT_LIM,
                            [0, 1], [0.])
all_interp_values = get_all_interpolating_values(approx_hopon_outhorizon_solver_augmented.interp)
all_interp_states = get_all_interpolating_points(approx_hopon_outhorizon_solver_augmented.interp)

outhor_interp_values = Vector{Float64}()

# copy over the values for horizon = K
for (v,s) in zip(all_interp_values,all_interp_states)
  if s[end] == 0.
    push!(outhor_interp_values,v)
  end
end

@assert length(outhor_interp_values) == length(hopon_grid_outhor)
out_hor_approx_true = LocalGIFunctionApproximator(hopon_grid_outhor,outhor_interp_values)
approx_hopon_outhorizon_policy = LocalApproximationValueIterationPolicy(out_hor_approx_true,ordered_actions(pc_hopon_mdp),
                                                                      pc_hopon_mdp, approx_hopon_outhorizon_solver_augmented.is_mdp_generative,
                                                                      approx_hopon_outhorizon_solver_augmented.n_generative_samples)

# Now create full policy
hopon_policy = PartialControlHopOnPolicy(approx_hopon_inhorizon_policy, approx_hopon_outhorizon_policy, ordered_actions(pc_hopon_mdp))
save("hopon_policy_sigmapt_paramset1.jld","policy",hopon_policy)