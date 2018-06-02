mutable struct HopOnOffSingleCarSimulator
    timestep::Float64
    time_to_finish::Distributions.Normal{Float64}
    mdp::Union{ControlledMultiRotorHopOnMDP,HopOffMDP}
    rng::AbstractRNG
end

function HopOnOffSingleCarSimulator(timestep::Float64,mdp::Union{ControlledMultiRotorHopOnMDP,HopOffMDP},rng::AbstractRNG=Base.GLOBAL_RNG)
    return HopOnOffSingleCarSimulator(timestep,Distributions.Normal(0.0,CAR_TIME_STD),mdp,rng)
end

function reset_sim(sim::HopOnOffSingleCarSimulator,meanval::Float64,stdval::Float64)
    sim.time_to_finish = Distributions.Normal(meanval,stdval)
end

function reset_sim(sim::HopOnOffSingleCarSimulator)
    # randomly sample the initial time for car to reach waypoint
    val = rand(sim.rng,Uniform(0.5*MDP_TIMESTEP*HORIZON_LIM,1.5*MDP_TIMESTEP*HORIZON_LIM))
    sim.time_to_finish = Distributions.Normal(val,CAR_TIME_STD)
end

function sample_finish_time(sim::HopOnOffSingleCarSimulator)
    return rand(sim.rng,sim.time_to_finish)
end


function step_sim(sim::HopOnOffSingleCarSimulator, state::ControlledHopOnState, action::HopOnAction)

    # First generate new state for UAV
    new_state, reward = generate_sr(sim.mdp, state, action, sim.rng)

    # Now update car state
    old_mean = mean(sim.time_to_finish)
    car_std = std(sim.time_to_finish)

    # Sample either a delay 
    probval = rand(sim.rng)
    if probval < DELAY_SPEEDUP_PROB
        is_delay = -1
    elseif probval < 1.0 - DELAY_SPEEDUP_PROB
        is_delay = 0
    else
        is_delay = 1
    end

    if is_delay == -1
        # Delay
        delay::Float64 = rand(sim.rng,Uniform(0.0,MAX_DELAY_SPEEDUP))
        sim.time_to_finish = Distributions.Normal(old_mean+delay, car_std)
    elseif is_delay == 0
        # Normal
        sim.time_to_finish = Distributions.Normal(old_mean - sim.timestep, car_std)
    else
        # Speedup
        # Can't sample more than time left
        max_speedup_val = min(old_mean, MAX_DELAY_SPEEDUP)
        speedup::Float64 = rand(sim.rng,Uniform(0.0,max_speedup_val))

        if sim.timestep + speedup > old_mean
            sim.time_to_finish = Distributions.Normal(0.0,car_std)
        else
            sim.time_to_finish = Distributions.Normal(old_mean - (sim.timestep+speedup), car_std)
        end
    end

    # If too little time left, car is at end
    is_done::Bool =  (mean(sim.time_to_finish) < std(sim.time_to_finish))

    return new_state, reward, is_done
end