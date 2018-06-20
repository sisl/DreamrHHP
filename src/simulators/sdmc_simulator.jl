mutable struct SDMCState{US <: UAVState}
    uav_state::US
    on_car::Bool
    car_id::String
end

SDMCAction = Union{UA,Tuple{HOP_ACTION,String}} where {UA <: UAVAction}


mutable struct SDMCSimulator{RNG <: AbstractRNG, UDM <: UAVDynamicsModel}
    epochs_dict::Dict
    uav_dynamics::UDM
    state::SDMCState
    goal_pos::Point
    epoch_counter::Int64
    rng::RNG
end


function SDMCSimulator(epochs_dict::Dict, uav_dynamics::UDM, start_pos::Point, goal_pos::Point, rng::RNG=Base.GLOBAL_RNG) where {RNG <: AbstractRNG, UDM <: UAVDynamicsModel}

    # Get start state
    state::SDMCState = SDMCState(get_state_at_rest(uav_dynamics, start_pos), false, "")

    return SDMCSimulator(epochs_dict, uav_dynamics, state, goal_pos, 0, rng)
end

function get_epoch0_dict(sdmc::SDMCSimulator)
    return sdmc.epochs_dict["0"]
end


function step_SDMC(sdmc::SDMCSimulator, action::SDMCAction)

    # Increase Counter
    sdmc.epoch_counter += 1

    # Read in next epoch
    epoch_info_dict = sdmc.epochs_dict[string(sdmc.epoch_counter)]
    epoch_car_info = epoch_info_dict["car-info"]

    reward::Float64 = 0.0

    # If UAV action, simulate and add reward and return car dict as additional info (OpenAIGYM style)
    if typeof(action) <: UAVAction
        new_uavstate = next_state(sdmc.uav_dynamics, sdmc.state.uav_state, action, sdmc.rng)
        reward += -dynamics_cost(sdmc.uav_dynamics, sdmc.state.uav_state, new_uavstate)
        sdmc.state.uav_state = new_uavstate # Update system state
    else
        if action[1] == HOPON

            # Must not be on a car currently
            if sdmc.state.on_car == true
                warn("Cannot Hop On while on a car!")
                reward += -INVALID_ACTION_PENALTY
            else
                hopon_car_id = action[2]

                car_pos::Point = Point(epoch_car_info[hopon_car_id]["pos"][1],epoch_car_info[hopon_car_id]["pos"][2])
                uav_pos::Point = Point(sdmc.state.uav_state.x, sdmc.state.uav_state.y)

                if point_dist(car_pos, uav_pos) < 2*MDP_TIMESTEP*HOP_DISTANCE_THRESHOLD
                    warn("Successful hop on to ",hopon_car_id, "at epoch ",sdmc.epoch_counter)
                    curr_car_pos = Point(epoch_car_info[hopon_car_id]["pos"][1],epoch_car_info[hopon_car_id]["pos"][2])
                    sdmc.state.uav_state = get_state_at_rest(sdmc.uav_dynamics, curr_car_pos)
                    sdmc.state.on_car = true
                    sdmc.state.car_id = hopon_car_id
                else
                    warn("Too far from car to hop on!")
                    reward += -INVALID_ACTION_PENALTY
                end
            end
        elseif action[1] == STAY

            # Must be on a car currently
            if sdmc.state.on_car == false
                warn("Cannot Stay when not on car!")
                reward += -INVALID_ACTION_PENALTY
            else
                # Assign new location to uav
                current_car = sdmc.state.car_id
                curr_car_pos = Point(epoch_car_info[current_car]["pos"][1],epoch_car_info[current_car]["pos"][2])
                sdmc.state.uav_state = get_state_at_rest(sdmc.uav_dynamics, curr_car_pos)
            end

        else
            # Must be HOPOFF
            if sdmc.state.on_car == false
                warn("Cannot Hop Off when not on car!")
                reward += -INVALID_ACTION_PENALTY
            else
                warn("Successful hopoff from car ",sdmc.state.car_id, " at epoch ",sdmc.epoch_counter)
                current_car = sdmc.state.car_id
                curr_car_pos = Point(epoch_car_info[current_car]["pos"][1],epoch_car_info[current_car]["pos"][2])
                sdmc.state.uav_state = get_state_at_rest(sdmc.uav_dynamics, curr_car_pos)
                sdmc.state.on_car = false
                sdmc.state.car_id = ""
            end
        end
    end

    is_terminal::Bool = false
    # Check if at goal
    if point_dist(Point(sdmc.state.uav_state.x, sdmc.state.uav_state.y), sdmc.goal_pos) < 5*MDP_TIMESTEP*HOP_DISTANCE_THRESHOLD
        reward += SUCCESS_REWARD
        is_terminal = true
    end

    return sdmc.state, reward, is_terminal, epoch_info_dict

end