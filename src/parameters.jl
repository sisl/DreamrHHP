#= 
A collection of several problem parameters for use by various MDP models.
Collected here as a bunch of global variables with module-level scope
Assume all are SI units?
=#

struct ScaleParameters
    EPSILON::Float64
    ACCELERATION_LIM::Float64
    ACCELERATION_NUMVALS::Int
    HOP_DISTANCE_THRESHOLD::Float64
    XY_LIM::Float64
    XY_AXISVALS::Int
    XYDOT_LIM::Float64
    XYDOT_AXISVALS::Int
    XYDOT_HOP_THRESH::Float64
    HORIZON_LIM::Int
    ACC_NOISE_STD::Float64
    MAX_DRONE_SPEED::Float64
    MAX_CAR_SPEED::Float64
    VALID_FLIGHT_EDGE_DIST_RATIO::Float64
    MC_GENERATIVE_NUMSAMPLES::Int
end

struct SimTimeParameters
    MDP_TIMESTEP::Float64
    WAYPT_TIME_CHANGE_THRESHOLD::Float64
    MAX_REPLAN_TIMESTEP::Float64
    CAR_TIME_STD::Float64
    DELAY_SPEEDUP_PROB::Float64
    MAX_DELAY_SPEEDUP::Float64
    MC_TIME_NUMSAMPLES::Int
end

struct CostParameters
    FLIGHT_COEFFICIENT::Float64
    HOVER_COEFFICIENT::Float64
    TIME_COEFFICIENT::Float64
    NO_HOPOFF_PENALTY::Float64
    FLIGHT_REACH_REWARD::Float64
end



struct Parameters
    scale_params::Union{ScaleParameters, Nothing}
    time_params::Union{SimTimeParameters, Nothing}
    cost_params::Union{CostParameters, Nothing}
end


function parse_scale(filename::AbstractString)

    params_key = TOML.parsefile(filename)

    return ScaleParameters(params_key["EPSILON"],
                           params_key["ACCELERATION_LIM"],
                           params_key["ACCELERATION_NUMVALS"],
                           params_key["HOP_DISTANCE_THRESHOLD"],
                           params_key["XY_LIM"],
                           params_key["XY_AXISVALS"],
                           params_key["XYDOT_LIM"],
                           params_key["XYDOT_AXISVALS"],
                           params_key["XYDOT_HOP_THRESH"],
                           params_key["HORIZON_LIM"],
                           params_key["ACC_NOISE_STD"],
                           params_key["MAX_DRONE_SPEED"],
                           params_key["MAX_CAR_SPEED"],
                           params_key["VALID_FLIGHT_EDGE_DIST_RATIO"],
                           params_key["MC_GENERATIVE_NUMSAMPLES"])
end


function parse_simtime(filename::AbstractString)

    params_key = TOML.parsefile(filename)

    return SimTimeParameters(params_key["MDP_TIMESTEP"],
                             get(params_key, "WAYPT_TIME_CHANGE_THRESHOLD", params_key["MDP_TIMESTEP"]/2.0),
                             params_key["MAX_REPLAN_TIMESTEP"],
                             get(params_key, "CAR_TIME_STD", params_key["MDP_TIMESTEP"]/2.0),
                             params_key["DELAY_SPEEDUP_PROB"],
                             params_key["MC_TIME_NUMSAMPLES"])
end


function parse_cost(filename::AbstractString)

    params_key = TOML.parsefile(filename)

    return CostParameters(params_key["FLIGHT_COEFFICIENT"],
                          params_key["HOVER_COEFFICIENT"],
                          params_key["TIME_COEFFICIENT"],
                          params_key["NO_HOPOFF_PENALTY"],
                          params_key["FLIGHT_REACH_REWARD"])
end


function parse_params(;scale_file::AbstractString="", simtime_file::AbstractString="", cost_file::AbstractString="")

    spar = (scale_file != "") ? parse_scale(scale_file) : nothing
    stpar = (simtime_file != "") ? parse_simtime(simtime_file) : nothing
    cpar = (cost_file != "") ? parse_cost(cost_file) : nothing

    return Parameters(spar, stpar, cpar)
end