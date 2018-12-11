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
using Statistics
using HitchhikingDrones

rng = MersenneTwister(15)

# Triage for now - inhor and outhor policy names
inhor_policy_names = ["../data/policies/test-cf-poly-abort_thresh-0.5-inhor.jld2"]
outhor_policy_names = ["../data/policies/test-cf-poly-preabort-outhor.jld2"]


# Comment out when running script
ARGS = ["../data/paramsets/scale-small-test.toml","../data/paramsets/simtime-small-test.toml",
        "../data/paramsets/cost-1.toml",
        "test-smallscale.txt", "1", "0.75"]

scale_file = ARGS[1]
simtime_file = ARGS[2]
cost_file = ARGS[3]
outfilename = ARGS[4]
NUM_EPISODES = parse(Int64,ARGS[5])
energy_time_alpha = parse(Float64, ARGS[6])

# First, parse parameter files to get filenames and construct params object
params = parse_params(scale_file=scale_file, simtime_file=simtime_file, cost_file=cost_file)

# Now create MDP and simulator
uav_dynamics = MultiRotorUAVDynamicsModel(params.time_params.MDP_TIMESTEP, params.scale_params.ACC_NOISE_STD, params)
pc_hopon_mdp = ControlledHopOnMDP{MultiRotorUAVState, MultiRotorUAVAction}(uav_dynamics, energy_time_alpha)

sim = HopOnOffSingleCarSimulator(params.time_params.MDP_TIMESTEP, params, rng)

outfile = open(outfilename,"w")

for (inhor_pn, outhor_pn) in zip(inhor_policy_names, outhor_policy_names)

    @show inhor_pn
    @show outhor_pn

    write(outfile,string(inhor_pn,"\t",outhor_pn,"\n"))
    
    hopon_policy = load_partialcontrolpolicy(inhor_pn, outhor_pn)

    rewards = Vector{Float64}(undef, NUM_EPISODES)
    successes = Vector{Int64}(undef, NUM_EPISODES)
    aborts = 0

    avg_diff_aborts = 0.0
    avg_diff_success_val_nom = 0.0
    avg_diff_success_true_nom = 0.0

    for i = 1:NUM_EPISODES

        reset_sim(sim)
        reward = 0.0
        is_success = false
        is_abort = false

        curr_uavstate = generate_start_state(uav_dynamics, rng)

        println("Episode - ",i)

        start_value = 0.0
        nominal_cost = 0.0
        start_pos = Point()
        start_time = 0.0
        log_value = false
        aug_inhor_state_temp = nothing

        while true

            time_to_finish_prob = zeros(params.time_params.HORIZON_LIM+2)
            for j = 1:params.time_params.MC_TIME_NUMSAMPLES
                tval::Float64 = sample_finish_time(sim)/params.time_params.MDP_TIMESTEP
                if tval >= params.time_params.HORIZON_LIM
                    time_to_finish_prob[end] += 1.0
                else
                    # Tricky part - don't round off but rather allocate
                    low = convert(Int64,floor(tval))
                    high = convert(Int64,ceil(tval))
                    low_wt = tval - floor(tval)
                    time_to_finish_prob[max(1,low+1)] += low_wt
                    time_to_finish_prob[max(1,high+1)] += 1.0 - low_wt
                end
            end


            # Normalize
            @assert sum(time_to_finish_prob) > 0.0
            time_to_finish_prob /= sum(time_to_finish_prob)

            tf = mean(sim.time_to_finish)
            start_pos = Point(curr_uavstate.x, curr_uavstate.y)
            if tf < params.time_params.HORIZON_LIM*params.time_params.MDP_TIMESTEP && log_value == false
                # log value first time
                log_value = true
                # time_est = 0.0
                # # start_value = 0.0
                # for hor = 2:HORIZON_LIM
                #     time_prob = time_to_finish_prob[hor+1]

                #     if time_prob > 0.0
                start_time = tf
                hor = convert(Int64,floor(tf/params.time_params.MDP_TIMESTEP))
                aug_inhor_state_temp = ControlledHopOnStateAugmented{MultiRotorUAVState}(curr_uavstate, hor)
                start_value = value(hopon_policy.in_horizon_policy, aug_inhor_state_temp)
                #         time_est += time_prob*MDP_TIMESTEP*hor 
                #     end
                # end

                # nominal_cost
                nominal_cost = params.cost_params.FLIGHT_COEFFICIENT*point_norm(Point(curr_uavstate.x, curr_uavstate.y)) + 
                               params.cost_params.TIME_COEFFICIENT*tf
            end


            action_values = zeros(n_actions(pc_hopon_mdp))

            @info "Iterating over actions"
            # Now create a ghost state for each time_to_finish_prob
            for a in actions(pc_hopon_mdp)
                iaction = actionindex(pc_hopon_mdp,a)

                # For horizon 0, just lookup value regardless of action

                # For horizon 1 to HOR_LIM, augment state with horizon and lookup action value from in_hor policy
                # Then add based on weight in time_to_finish_prob
                for hor = 1:params.time_params.HORIZON_LIM
                    time_prob = time_to_finish_prob[hor+1]

                    if time_prob > 0.0
                        aug_inhor_state = ControlledHopOnStateAugmented{MultiRotorUAVState}(curr_uavstate, hor)
                        action_values[iaction] += time_prob*action_value(hopon_policy.in_horizon_policy, aug_inhor_state, a)
                    end
                end

                # For horizon HOR_LIM + 1, augment state with horizon 0 and lookup action value from out_hor policy
                time_prob = time_to_finish_prob[end]
                if time_prob > 0.0
                    aug_outhor_state = ControlledHopOnStateAugmented{MultiRotorUAVState}(curr_uavstate, params.time_params.HORIZON_LIM+1)
                    action_values[iaction] += time_prob*action_value(hopon_policy.out_horizon_policy, aug_outhor_state, a)
                end
            end # for a

            # Choose best action
            # From EITHER action_map
            best_action_idx = argmax(action_values)
            best_action = hopon_policy.in_horizon_policy.action_map[best_action_idx]


            @show best_action

            # Step forward and add reward
            if best_action.control_transfer == true
                println("ABORTED!")
                is_abort = true
                aborts += 1
                break
            end

            # Dynamics action
            cost = params.time_params.TIME_COEFFICIENT*params.time_params.MDP_TIMESTEP
            new_uavstate = next_state(uav_dynamics, curr_uavstate, best_action.uavaction, rng)
            cost += dynamics_cost(uav_dynamics, curr_uavstate, new_uavstate)

            reward += -cost

            is_done = step_sim(sim)
            @show is_done

            curr_uavstate = new_uavstate
            
            # Break if done
            if is_done
                curr_rel_pos = Point(curr_uavstate.x, curr_uavstate.y)
                curr_speed = sqrt(curr_uavstate.xdot^2 + curr_uavstate.ydot^2)
                if point_norm(curr_rel_pos) < params.time_params.MDP_TIMESTEP*params.scale_params.HOP_DISTANCE_THRESHOLD && 
                    curr_speed < params.scale_params.XYDOT_HOP_THRESH
                    is_success = true
                end
                break
            end

            print("Press something to continue")
            readline()

        end # while

        if is_success
            avg_diff_success_true_nom += -reward - nominal_cost
            avg_diff_success_val_nom += -start_value - nominal_cost
        end

        if is_abort
            avg_diff_aborts += -reward - nominal_cost
        end

        # Accumulate reward and succcess
        rewards[i] = reward
        successes[i] = is_success

        println("Success : ",is_success,"| True cost - ",-reward,"| Nominal cost - ",nominal_cost,"| Start value - ",-start_value)
        println("Start rel pos - ",start_pos,"| Start time diff - ",start_time)

        # print("Press something to continue")
        # readline()
    end # for i

    avg_diff_success_val_nom = avg_diff_success_val_nom/sum(successes)
    avg_diff_success_true_nom = avg_diff_success_true_nom/sum(successes)
    avg_diff_aborts = avg_diff_aborts/aborts

    write(outfile, string("For successes: \n"))
    write(outfile,string("Start Cost - Nominal Cost = ",avg_diff_success_val_nom,"\n"))
    write(outfile,string("True Cost - Nominal Cost = ",avg_diff_success_true_nom,"\n"))

    write(outfile, string("For aborts: \n"))
    write(outfile,string("Start Cost - Nominal Cost = ",avg_diff_aborts,"\n"))

    write(outfile, string("AVG REWARDS - ",mean(rewards)),"\n")
    write(outfile, string("SUCCESS RATE - ",mean(successes)),"\n")
    write(outfile, string("ABORTS - ",aborts),"\n\n")
end

close(outfile)
