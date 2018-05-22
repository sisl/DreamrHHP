#= 
A collection of several problem parameters for use by various MDP models.
Collected here as a bunch of global variables with module-level scope
Assume all are SI units?
=#
# Dynamics parameters
const EPSILON = 0.0000001
const ACCELERATION_VALUES = linspace(-1.0,1.0,5) # M / s^2
const DISTANCE_THRESHOLD = 1.0
const XY_LIM = 30.0
const XY_RES = 10.0
const XYDOT_LIM = 4.0
const XYDOT_RES = 1.0
const HORIZON_LIM = 20
const ACC_NOISE_STD = 0.1
const MAX_SPEED = 20.0

# Sigma point weights
const SIGMA_PT_ALPHA = 1.0
const SIGMA_PT_KAPPA = 1.0

# System freq parameters
const MDP_TIMESTEP = 5.0
const OPENLOOP_DURATION = 30.0


# Cost parameters
const FLIGHT_COEFFICIENT = 1.0
const TIME_COEFFICIENT = 0.5
const HOP_COEFFICIENT = 10.0
const HOVER_COEFFICIENT = 1.0
const HOP_REWARD = 100.0
const CONTROL_TRANSFER_PENALTY = 100.0
const INVALID_ACTION_PENALTY = 10.0 # For the real simulator
const SUCCESS_REWARD = 1000.0 # For the real simulator
const VALID_FLIGHT_EDGE_DIST_RATIO = 1.2 # The minimum ratio between dist of a flight edge and (max_speed*time_diff)

# Additional simulator parameters - UNKNOWN to agent
const CAR_TIME_STD = MDP_TIMESTEP/2.0
const DELAY_SPEEDUP_PROB = 0.2
const MAX_DELAY_SPEEDUP = 2.0*MDP_TIMESTEP