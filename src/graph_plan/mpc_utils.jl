function get_flight_mpc_action_multirotor(curr_state::MultiRotorUAVState, 
                                          next_vertex::CarDroneVertex, 
                                          curr_fin_horizon::Int64)

    # MPC timesteps depends on finish horizon
    N = curr_fin_horizon > 20 ? 20 : curr_fin_horizon


    # Setup solver
    solver = IpoptSolver(linear_solver="mumps",max_iter=100,print_level=0)

    # Obtain start and goal in terms of curr_state and next_vertex
    curr_vect = SVector{4, Float64}(curr_state.x, curr_state.y, curr_state.xdot, curr_state.ydot)
    curr_goal_pos = SVector{2,Float64}(next_vertex.pos.x, next_vertex.pos.y)

    m = JuMP.Model(solver = solver)

    @variable(m, uav_state[1:4*(N+1)])

    for i in 1:N+1
        setlowerbound(uav_state[4*(i-1)+3], -XYDOT_LIM)
        setlowerbound(uav_state[4*(i-1)+4], -XYDOT_LIM)
        setupperbound(uav_state[4*(i-1)+3], XYDOT_LIM)
        setupperbound(uav_state[4*(i-1)+4], XYDOT_LIM)
    end

    # Set start state
    @constraint(m, uav_state[1:4] .== curr_vect)

    # Set end velocity
    @NLconstraint(m, abs(uav_state[4*N+3]^2) <= XYDOT_HOP_THRESH/sqrt(2))
    @NLconstraint(m, abs(uav_state[4*N+4]^2) <= XYDOT_HOP_THRESH/sqrt(2))

    # Define control actions
    @variable(m, -ACCELERATION_LIM <= acc[1:2*N] <= ACCELERATION_LIM)

    # Setup objective function
    @NLobjective(m, Min, sum(FLIGHT_COEFFICIENT*sqrt( (curr_goal_pos[1] - uav_state[4*(i-1)+1])^2 + 
        (curr_goal_pos[2] - uav_state[4*(i-1)+2])^2) 
        + HOVER_COEFFICIENT*( sqrt(uav_state[4*(i-1) + 3]^2 + uav_state[4*(i-1) + 4]^2)  < MDP_TIMESTEP*EPSILON)
         for i = 1:N+1))

    # Control Constraint
    for i in 1:N
        # x_new = x_old + xdot_old*t + 0.5*a*t^2
        @constraint(m, uav_state[4*i+1]-uav_state[4*(i-1)+1] - (uav_state[4*(i-1)+3]*MDP_TIMESTEP + 
            0.5*acc[2*(i-1)+1]*MDP_TIMESTEP^2) == 0)
        @constraint(m, uav_state[4*i+2]-uav_state[4*(i-1)+2] - (uav_state[4*(i-1)+4]*MDP_TIMESTEP + 
            0.5*acc[2*i]*MDP_TIMESTEP^2) == 0)

        # xdot_new = xdot_old + a*t
        @constraint(m, uav_state[4*i+3]-uav_state[4*(i-1)+3] - acc[2*(i-1)+1]*MDP_TIMESTEP == 0)
        @constraint(m, uav_state[4*i+4]-uav_state[4*(i-1)+4] - acc[2*i]*MDP_TIMESTEP == 0)
    end

    status = JuMP.solve(m)

    accvals = getvalue(acc)
    xddotvals = accvals[1:2:end]
    yddotvals = accvals[2:2:end]

    curr_action = MultiRotorUAVAction(xddotvals[1], yddotvals[1])

    return curr_action
end

