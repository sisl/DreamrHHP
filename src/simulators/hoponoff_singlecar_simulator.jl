## NOTE - This is an adhoc simulator that was only used for testing the constrained flight behavior.
mutable struct HopOnOffSingleCarSimulator
    timestep::Float64
    time_to_finish::Distributions.Normal{Float64}
    params::Parameters
    rng::AbstractRNG
end

function HopOnOffSingleCarSimulator(timestep::Float64, params::Parameters, rng::AbstractRNG=Random.GLOBAL_RNG)
    return HopOnOffSingleCarSimulator(timestep, Distributions.Normal(0.0,params.time_params.CAR_TIME_STD), params, rng)
end

function reset_sim(sim::HopOnOffSingleCarSimulator,meanval::Float64,stdval::Float64)
    sim.time_to_finish = Distributions.Normal(meanval,stdval)
end

function reset_sim(sim::HopOnOffSingleCarSimulator)
    # randomly sample the initial time for car to reach waypoint
    val = rand(sim.rng,Uniform(0.5*sim.params.time_params.MDP_TIMESTEP*sim.params.time_params.HORIZON_LIM,
                               1.25*sim.params.time_params.MDP_TIMESTEP*sim.params.time_params.HORIZON_LIM))
    sim.time_to_finish = Distributions.Normal(val,sim.params.time_params.CAR_TIME_STD)
end

function sample_finish_time(sim::HopOnOffSingleCarSimulator)
    return rand(sim.rng,sim.time_to_finish)
end


function step_sim(sim::HopOnOffSingleCarSimulator) 

    # These params are just for testing simulator
    DELAY_SPEEDUP_PROB = 0.4
    MAX_DELAY_SPEEDUP = params.time_params.MDP_TIMESTEP*2.0

    old_mean = mean(sim.time_to_finish)
    car_std = std(sim.time_to_finish)

    # Split 1 into 3 bins - 0.0: delay_speedup_prob/2 : 1.0-dsp/2 : 1.0
    probval = rand(sim.rng)
    if probval < DELAY_SPEEDUP_PROB/2.0
        is_delay = -1
    elseif probval < 1.0 - DELAY_SPEEDUP_PROB/2.0
        is_delay = 0
    else # probval > 1.0 - dsp/2.0
        is_delay = 1
    end

    if is_delay == -1
        # Delay
        delay = rand(sim.rng,Uniform(0.0,MAX_DELAY_SPEEDUP))
        sim.time_to_finish = Distributions.Normal(old_mean+delay, car_std)
    elseif is_delay == 0
        # Normal
        sim.time_to_finish = Distributions.Normal(old_mean - sim.timestep, car_std)
    else
        # Speedup
        # Can't sample more than time left
        max_speedup_val = min(old_mean, MAX_DELAY_SPEEDUP)
        speedup = rand(sim.rng,Uniform(0.0,max_speedup_val))

        if sim.timestep + speedup > old_mean
            sim.time_to_finish = Distributions.Normal(0.0,car_std)
        else
            sim.time_to_finish = Distributions.Normal(old_mean - (sim.timestep+speedup), car_std)
        end
    end

    # If too little time left, car is at end
    is_done =  (mean(sim.time_to_finish) < std(sim.time_to_finish))

    return is_done
end