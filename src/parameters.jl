#= 
A collection of several problem parameters for use by various MDP models.
Collected here as a bunch of global variables with module-level scope
Assume all are SI units?
=#
# Dynamics parameters
# REAL-WORLD
# const EPSILON = 0.00001
# const ACCELERATION_LIM = 1.0
# const ACCELERATION_VALS = 7
# const HOP_DISTANCE_THRESHOLD = 1.0
# const XY_LIM = 30.0
# const XY_AXISVALS = 7
# const XYDOT_LIM = 4.0
# const XYDOT_AXISVALS = 9
# const HORIZON_LIM = 20
# const ACC_NOISE_STD = 0.1
# const MAX_DRONE_SPEED = 5.6
# const MAX_CAR_SPEED = 20.0


# Scale Params
const EPSILON = 0.00001
const ACCELERATION_LIM = 0.0002
const ACCELERATION_NUMVALS = 7
const HOP_DISTANCE_THRESHOLD = 0.001
const XY_LIM = 1.0
const XY_AXISVALS = 19
const XYDOT_LIM = 0.002
const XYDOT_AXISVALS = 9
const XYDOT_HOP_THRESH = 0.0008
const HORIZON_LIM = 100
const ACC_NOISE_STD = 0.00001
const MAX_DRONE_SPEED = 0.00285
const MAX_CAR_SPEED = 0.007
const VALID_FLIGHT_EDGE_DIST_RATIO = 1.1 # The minimum ratio between dist of a flight edge and (max_speed*time_diff)


# Time Params
const MDP_TIMESTEP = 5.0
const WAYPT_TIME_CHANGE_THRESHOLD = MDP_TIMESTEP/2.0
const MAX_REPLAN_TIMESTEP = 20.0


# Cost Params
const FLIGHT_COEFFICIENT = 5000.0
const HOVER_COEFFICIENT = 100.0
const TIME_COEFFICIENT = 1.0
const NO_HOPOFF_PENALTY = 1000
const FLIGHT_REACH_REWARD = 10000

# Additional simulator params - UNKNOWN to agent
const CAR_TIME_STD = MDP_TIMESTEP/2.0
const DELAY_SPEEDUP_PROB = 0.2
const MAX_DELAY_SPEEDUP = 2.0*MDP_TIMESTEP
const MC_TIME_NUMSAMPLES = 20
const MC_GENERATIVE_NUMSAMPLES = 20



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

struct TimeParameters
    MDP_TIMESTEP::Float64
    WAYPT_TIME_CHANGE_THRESHOLD::Float64
    MAX_REPLAN_TIMESTEP::Float64
end

struct CostParameters
    FLIGHT_COEFFICIENT::Float64
    HOVER_COEFFICIENT::Float64
    TIME_COEFFICIENT::Float64
    NO_HOPOFF_PENALTY::Float64
    FLIGHT_REACH_REWARD::Float64
end

struct SimParameters
    CAR_TIME_STD::Float64
    DELAY_SPEEDUP_PROB::Float64
    MAX_DELAY_SPEEDUP::Float64
    MC_TIME_NUMSAMPLES::Int
end


struct Parameters
    scale_params::Union{ScaleParameters, Nothing}
    time_params::Union{TimeParameters, Nothing}
    cost_params::Union{CostParameters, Nothing}
    sim_params::Union{SimParameters, Nothing}
end

# Parse for any one aspect or for all aspects
parse_params()
parse_params()
parse_params()
parse_params()
parse_params()