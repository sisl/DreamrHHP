using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration
using POMDPs
using POMDPModels
using POMDPToolbox
using StaticArrays
using JLD
using Distributions

using HitchhikingDrones

# LOAD POLICY HERE - SWAP OUT AS NEEDED
policy_names = [ "policies/hopon_denseparamset1-exp-abort_thresh-0.55.jld",
                "policies/hopon_denseparamset1-exp-abort_thresh-0.75.jld",
                "policies/hopon_denseparamset1-exp-abort_thresh-0.95.jld"]

outfilename = "policies_compared_2.txt"

outfile = open(outfilename,"w")


rng = MersenneTwister(1)
NUM_EPISODES = 20

# Now create MDP and simulator
uav_dynamics = MultiRotorUAVDynamicsModel(MDP_TIMESTEP, ACC_NOISE_STD)
pc_hopon_mdp = ControlledMultiRotorHopOnMDP(uav_dynamics)


sim = HopOnOffSingleCarSimulator(MDP_TIMESTEP,rng)



for pn in policy_names

    println(pn)
    write(outfile,string(pn,"\n"))
    hopon_policy = load(pn,"hopon_policy")

    rewards = Vector{Float64}(NUM_EPISODES)
    successes = Vector{Int64}(NUM_EPISODES)
    aborts::Int64 = 0

    for i = 1:NUM_EPISODES

        reset_sim(sim)
        reward::Float64 = 0.0
        is_success::Int64 = 0

        curr_uavstate = generate_start_state(uav_dynamics, rng)

        println("Episode - ",i)

        while true


            # println(sim.time_to_finish)
            # println(curr_uavstate)

            time_to_finish_prob = zeros(HORIZON_LIM+2)
            for j = 1:MC_TIME_NUMSAMPLES
                tval::Float64 = sample_finish_time(sim)/MDP_TIMESTEP
                if tval >= HORIZON_LIM
                    time_to_finish_prob[end] += 1.0
                else
                    # Tricky part - don't round off but rather allocate
                    low = Int64(floor(tval))
                    high = Int64(ceil(tval))
                    low_wt = tval - floor(tval)
                    time_to_finish_prob[max(1,low+1)] += low_wt
                    time_to_finish_prob[max(1,high+1)] += 1.0 - low_wt
                end
            end

            # println(time_to_finish_prob)

            # Normalize
            @assert sum(time_to_finish_prob) > 0.0
            time_to_finish_prob /= sum(time_to_finish_prob)

            action_values = zeros(n_actions(pc_hopon_mdp))

            # Now create a ghost state for each time_to_finish_prob
            for a in iterator(actions(pc_hopon_mdp))
                iaction = action_index(pc_hopon_mdp,a)

                # For horizon 0, just lookup value regardless of action

                # For horizon 1 to HOR_LIM, augment state with horizon and lookup action value from in_hor policy
                # Then add based on weight in time_to_finish_prob
                for hor = 1:HORIZON_LIM
                    time_prob = time_to_finish_prob[hor+1]

                    if time_prob > 0.0
                        aug_inhor_state::ControlledHopOnStateAugmented = ControlledHopOnStateAugmented(curr_uavstate, hor)
                        action_values[iaction] += time_prob*action_value(hopon_policy.in_horizon_policy, aug_inhor_state, a)
                    end
                end

                # For horizon HOR_LIM + 1, augment state with horizon 0 and lookup action value from out_hor policy
                time_prob = time_to_finish_prob[end]
                if time_prob > 0.0
                    aug_outhor_state::ControlledHopOnStateAugmented = ControlledHopOnStateAugmented(curr_uavstate, HORIZON_LIM+1)
                    action_values[iaction] += time_prob*action_value(hopon_policy.out_horizon_policy,aug_outhor_state,a)
                end
            end # for a

            # Choose best action
            # From EITHER action_map
            best_action_idx::Int64 = indmax(action_values)
            best_action::HopOnAction = hopon_policy.in_horizon_policy.action_map[best_action_idx]


            #println("Action chosen - ",best_action)

            # Step forward and add reward
            if best_action.control_transfer == true
                #println("ABORTED!")
                aborts += 1
                break
            end

            # Dynamics action
            cost = TIME_COEFFICIENT*MDP_TIMESTEP
            new_uavstate = next_state(uav_dynamics, curr_uavstate, best_action.uavaction, rng)
            cost += dynamics_cost(uav_dynamics, curr_uavstate, new_uavstate)

            reward += -cost

            is_done = step_sim(sim)

            curr_uavstate = new_uavstate
            
            # Break if done
            if is_done
                curr_rel_pos = Point(curr_uavstate.x, curr_uavstate.y)
                curr_speed = sqrt(curr_uavstate.xdot^2 + curr_uavstate.ydot^2)
                if point_norm(curr_rel_pos) < HOP_DISTANCE_THRESHOLD && curr_speed < XYDOT_HOP_THRESH
                    is_success = 1
                end
                break
            end

            # print("Press something to continue")
            # readline()

        end # while

        # Accumulate reward and succcess
        rewards[i] = reward
        successes[i] = is_success

        # println(reward," ; ",is_success)

        # print("Press something to continue")
        # readline()

    end # for i


    # Print final results
    # println("AVG REWARDS - ",mean(rewards))
    # println("SUCCESS RATE - ",mean(successes))
    # println("ABROTS - ",aborts)

    write(outfile, string("AVG REWARDS - ",mean(rewards)),"\n")
    write(outfile, string("SUCCESS RATE - ",mean(successes)),"\n")
    write(outfile, string("ABORTS - ",aborts),"\n\n")
end

close(outfile)