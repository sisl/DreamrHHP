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


# UNIT GRID
const EPSILON = 0.0000001
const ACCELERATION_LIM = 0.0002
const ACCELERATION_NUMVALS = 7
const HOP_DISTANCE_THRESHOLD = 0.001
const XY_LIM = 1.0
const XY_AXISVALS = 17
const XYDOT_LIM = 0.002
const XYDOT_AXISVALS = 9
const XYDOT_HOP_THRESH = 0.0002 
const HORIZON_LIM = 100
const ACC_NOISE_STD = 0.00001
const MAX_DRONE_SPEED = 0.00285
const MAX_CAR_SPEED = 0.005

# Sigma point weights
const SIGMA_PT_ALPHA = 1.0
const SIGMA_PT_KAPPA = 1.0

# System freq parameters
const MDP_TIMESTEP = 5.0
const WAYPT_TIME_CHANGE_THRESHOLD = MDP_TIMESTEP/2.0


# Cost parameters
const FLIGHT_COEFFICIENT = 1000.0
const TIME_COEFFICIENT = 0.5
const HOVER_COEFFICIENT = 10.0
const HOP_REWARD = 1800.0 # High enough that in average circumstances it does not abort = 1000*sqrt(2) + 1*5*HORIZON_LIM 
const CONTROL_TRANSFER_PENALTY = 100.0
const INVALID_ACTION_PENALTY = 10.0 # For the real simulator
const VALID_FLIGHT_EDGE_DIST_RATIO = 1.25 # The minimum ratio between dist of a flight edge and (max_speed*time_diff)

# Additional simulator parameters - UNKNOWN to agent
const CAR_TIME_STD = MDP_TIMESTEP/2.0
const DELAY_SPEEDUP_PROB = 0.2
const MAX_DELAY_SPEEDUP = 2.0*MDP_TIMESTEP
const MC_TIME_NUMSAMPLES = 20
const MC_GENERATIVE_NUMSAMPLES = 20