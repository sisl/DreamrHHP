using JSON


mutable struct SDMCState{US <: UAVState}
    uav_state::US
    on_car::Bool
    on_car_id::String
end

SDMCAction = Union{UA,Tuple{HOP_ACTION,String}} where {UA <: UAVAction}


mutable struct SDMCSimulator{RNG <: AbstractRNG, UDM <: UAVDynamicsModel}
    episode_dict::Dict
    uav_dynamics::UDM
    start_pos::Point
    goal_pos::Point
    epoch_counter::Int64
    episode_length::Int64
    rng::RNG
end


function SDMCSimulator(ep_filename::String, uav_dynamics::UDM, rng::RNG=Base.GLOBAL_RNG) where {RNG <: AbstractRNG, UDM <: UAVDynamicsModel}

    # Read in file as Dict
    ep_dict = Dict()

    open(ep_filename,"r") do f_ep
        # global ep_dict
        dicttxt = readstring(f_ep)
        ep_dict = JSON.parse(dicttxt)
    end

    # Obtain start and goal positions from dict
    # TODO : Modify as needed
    start_pos::Point = Point(ep_dict["start"][1], ep_dict["start"][2])
    goal_pos::Point = Point(ep_dict["goal"][1], ep_dict["goal"][2])

    # Get episode length


    return SDMCSimulator(ep_dict, uav_dynamics, start_pos, goal_pos, rng)
end


function step_SDMC(sdmc::SDMCSimulator, state::SDMCState, action::SDMCAction)

    # Increase Counter


    # Read in next epoch



    # If UAV action, simulate and add reward and return car dict as additional info (OpenAIGYM style)



    # If stay, hop on, hop off, test out and generate state for UAV accordingly (by looking up dict!)

end