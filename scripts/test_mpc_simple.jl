using HitchhikingDrones
using StaticArrays
using JuMP, Ipopt

uav_dynamics = MultiRotorUAVDynamicsModel(MDP_TIMESTEP, ACC_NOISE_STD, HOVER_COEFFICIENT, FLIGHT_COEFFICIENT)
flight_mdp = UnconstrainedFlightMDP(uav_dynamics, 1.0)

solver = IpoptSolver(linear_solver="mumps",max_iter=100,print_level=0)

N = 20

energy_time_alpha = 0.5

# Define start and goal
start_vect = SVector{4,Float64}(0.0,0.0,0.0,0.0)
curr_vect = SVector{4,Float64}(0.0,0.0,0.0,0.0)
goal_pos = [0.53,0.33]

while true
    m = Model(solver = solver)
    # Define entire state vector
    @variable(m, uav_state[1:4*(N+1)])
    # Constrain vel to be in limits
    for i in 1:N+1
        setlowerbound(uav_state[4*(i-1)+3], -XYDOT_LIM)
        setlowerbound(uav_state[4*(i-1)+4], -XYDOT_LIM)
        setupperbound(uav_state[4*(i-1)+3], XYDOT_LIM)
        setupperbound(uav_state[4*(i-1)+4], XYDOT_LIM)
    end
    # Constrain start state
    @constraint(m, uav_state[1:4] .== curr_vect)

    # Set end velocity
    @NLconstraint(m, abs(uav_state[4*N+3]^2) <= XYDOT_HOP_THRESH/sqrt(2))
    @NLconstraint(m, abs(uav_state[4*N+4]^2) <= XYDOT_HOP_THRESH/sqrt(2))

    # Constrain goal state
    # @NLconstraint(m, abs(goal_pos[1] - uav_state[4*N+1]^2) <= 0.005/sqrt(2))
    # @NLconstraint(m, abs(goal_pos[2] - uav_state[4*N+2]^2) <= 0.005/sqrt(2))

    # Define control actions
    @variable(m, -ACCELERATION_LIM <= acc[1:2*N] <= ACCELERATION_LIM)

    # Setup objective function
    @NLobjective(m, Min, (1.0-energy_time_alpha)*sum(FLIGHT_COEFFICIENT*sqrt( (goal_pos[1] - uav_state[4*(i-1)+1])^2 + 
        (goal_pos[2] - uav_state[4*(i-1)+2])^2)
        + HOVER_COEFFICIENT*( sqrt(uav_state[4*(i-1) + 3]^2 + uav_state[4*(i-1) + 4]^2)  < MDP_TIMESTEP*EPSILON)
         for i = 1:N+1))

    # Control Constraint
    for i in 1:N
        # x_new = x_old + xdot_old*t + 0.5*a*t^2
        @NLconstraint(m, uav_state[4*i+1]-uav_state[4*(i-1)+1] - (uav_state[4*(i-1)+3]*MDP_TIMESTEP + 
            0.5*acc[2*(i-1)+1]*MDP_TIMESTEP^2) == 0)
        @NLconstraint(m, uav_state[4*i+2]-uav_state[4*(i-1)+2] - (uav_state[4*(i-1)+4]*MDP_TIMESTEP + 
            0.5*acc[2*i]*MDP_TIMESTEP^2) == 0)

        # xdot_new = xdot_old + a*t
        @NLconstraint(m, uav_state[4*i+3]-uav_state[4*(i-1)+3] - acc[2*(i-1)+1]*MDP_TIMESTEP == 0)
        @NLconstraint(m, uav_state[4*i+4]-uav_state[4*(i-1)+4] - acc[2*i]*MDP_TIMESTEP == 0)
    end

    status = solve(m)

    statevals = getvalue(uav_state)
    xvals = statevals[1:4:end]
    yvals = statevals[2:4:end]
    xdotvals = statevals[3:4:end]
    ydotvals = statevals[4:4:end]

    accvals = getvalue(acc)
    xddotvals = accvals[1:2:end]
    yddotvals = accvals[2:2:end]

    curr_state = convert_s(MultiRotorUAVState,curr_vect,flight_mdp)
    curr_action = MultiRotorUAVAction(xddotvals[1], yddotvals[1])

    println(curr_action)

    next_s = next_state(uav_dynamics, curr_state, curr_action)

    println(xvals)
    println(xdotvals)
    println(xddotvals)
    println(xddotvals[1]," : ",yddotvals[1])

    curr_vect = convert_s(SVector{4,Float64}, next_s, flight_mdp)

    println(curr_vect)

    if norm(curr_vect[1:2] - goal_pos) < 0.005 && norm(curr_vect[3:4]) < XYDOT_HOP_THRESH
        break
    end

    readline()

end
