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
hopon_policy = load("policies/hopon_generative_unitgrid_paramset3.jld","policy")

rng = MersenneTwister(2)
NUM_EPISODES = 10

# Now create MDP and simulator
uav_dynamics = MultiRotorUAVDynamicsModel(MDP_TIMESTEP, ACC_NOISE_STD)
pc_hopon_mdp = ControlledMultiRotorHopOnMDP(uav_dynamics)


sim = HopOnOffSingleCarSimulator(MDP_TIMESTEP,pc_hopon_mdp,rng)

rewards = Vector{Float64}(NUM_EPISODES)
successes = Vector{Int64}(NUM_EPISODES)

for i = 1:NUM_EPISODES
    reset_sim(sim)
    reward::Float64 = 0.0
    is_success::Int64 = 0

    curr_state::ControlledHopOnState = generate_start_state(pc_hopon_mdp,rng)

    println("Episode - ",i)

    while true


        println(sim.time_to_finish)
        println(curr_state)

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

        println(time_to_finish_prob)

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
                    aug_inhor_state::ControlledHopOnStateAugmented = ControlledHopOnStateAugmented(curr_state.rel_uavstate,
                                                                                             curr_state.control_transfer,hor)
                    action_values[iaction] += time_prob*action_value(hopon_policy.in_horizon_policy, aug_inhor_state, a)
                end
            end

            # For horizon HOR_LIM + 1, augment state with horizon 0 and lookup action value from out_hor policy
            time_prob = time_to_finish_prob[end]
            if time_prob > 0.0
                aug_outhor_state::ControlledHopOnStateAugmented = ControlledHopOnStateAugmented(curr_state.rel_uavstate,
                                                                                                curr_state.control_transfer,0.)
                action_values[iaction] += time_prob*action_value(hopon_policy.out_horizon_policy,aug_outhor_state,a)
            end
        end # for a

        # Choose best action
        # From EITHER action_map
        best_action_idx::Int64 = indmax(action_values)
        best_action::HopOnAction = hopon_policy.in_horizon_policy.action_map[best_action_idx]

        # best_action_idx = rand(rng,1:n_actions(pc_hopon_mdp))
        # best_action::HopOnAction = ordered_actions(pc_hopon_mdp)[best_action_idx]

        println("Action chosen - ",best_action)

        # Step forward and add reward
        curr_state, this_reward, is_done = step_sim(sim,curr_state,best_action)
        println("Reward obtained at this step - ",this_reward)
        reward += this_reward

        # Break if done
        if is_done
            curr_rel_pos = Point(curr_state.rel_uavstate.x, curr_state.rel_uavstate.y)
            if point_norm(curr_rel_pos) < HOP_DISTANCE_THRESHOLD
                reward += HOP_REWARD
                is_success = 1
            else
                reward -= HOP_REWARD
            end
            break
        end

        # If control transferred, also break
        if curr_state.control_transfer == true
            println("ABORTED!")
            reward -= CONTROL_TRANSFER_PENALTY
            break
        end

        print("Press something to continue")
        readline()

    end # while

    # Accumulate reward and succcess
    rewards[i] = reward
    successes[i] = is_success

    println(reward," ; ",is_success)

    # print("Press something to continue")
    # readline()

end # for i


# Print final results
println("REWARDS - ",rewards)
println("SUCCESSES - ",successes)