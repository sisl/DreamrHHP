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

# Sigma point weights
const SIGMA_PT_ALPHA = 1.0
const SIGMA_PT_KAPPA = 1.0

# System freq parameters
const MDP_TIMESTEP = 5.0
const WAYPT_TIME_CHANGE_THRESHOLD = MDP_TIMESTEP/2.0
const MAX_REPLAN_TIMESTEP = 20.0
const VALID_FLIGHT_EDGE_DIST_RATIO = 1.1 # The minimum ratio between dist of a flight edge and (max_speed*time_diff)


# Cost parameters
const FLIGHT_COEFFICIENT = 5000.0
const HOVER_COEFFICIENT = 50.0
const TIME_COEFFICIENT = 1.0
const NO_HOPOFF_PENALTY = 1000
const FLIGHT_REACH_REWARD = 10000



# Additional simulator parameters - UNKNOWN to agent
const CAR_TIME_STD = MDP_TIMESTEP/2.0
const DELAY_SPEEDUP_PROB = 0.2
const MAX_DELAY_SPEEDUP = 2.0*MDP_TIMESTEP
const MC_TIME_NUMSAMPLES = 20
const MC_GENERATIVE_NUMSAMPLES = 20
