#= 
A collection of several problem parameters for use by various MDP models.
Collected here as a bunch of global variables with module-level scope
Assume all are SI units?
=#
# Dynamics parameters
const EPSILON = 0.0000001
const ACCELERATION_VALUES = linspace(-1.0,1.0,5) # M / s^2
const HOP_DISTANCE_THRESHOLD = 1.0
const XY_LIM = 30.0
const XY_RES = 10.0
const XYDOT_LIM = 4.0
const XYDOT_RES = 1.0
const HORIZON_LIM = 20
const ACC_NOISE_STD = 0.1

# Sigma point weights
const SIGMA_PT_ALPHA = 1.0
const SIGMA_PT_KAPPA = 1.0

# System frequency issues (in seconds / hertz)
const MDP_TIMESTEP = 1.0
# const SERVER_TIME = 10


# Cost parameters
const FLIGHT_COEFFICIENT = 1.0
const TIME_COEFFICIENT = 0.5
const HOP_COEFFICIENT = 10.0
const HOVER_COEFFICIENT = 1.0
const HOP_REWARD = 100.0
const CONTROL_TRANSFER_PENALTY = 100.0

# Additional simulator parameters - UNKNOWN to agent
const CAR_TIME_STD = MDP_TIMESTEP/2.0
const DELAY_SPEEDUP_PROB = 0.2
const MAX_DELAY_SPEEDUP = 2.0*MDP_TIMESTEP