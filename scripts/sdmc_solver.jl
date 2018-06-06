using GridInterpolations
using POMDPs
using POMDPModels
using POMDPToolbox
using StaticArrays
using LocalFunctionApproximation
using LocalApproximationValueIteration
using JLD
using JSON
using HitchhikingDrones



# NOTE - If the macro action is at a stage outside the bounds of the drone, just fly on the st.line path
# towards the goal till it is within range




# Read in the episode filename from args
ep_file = "../data/test-ep.json"
hopon_file = "policies/hopon_generative_unitgrid_paramset2.jld"
hopoff_file = "policies/hopoff.jld"

episode_dict = Dict()
open(ep_file,"r") do f
    global episode_dict
    episode_dict = JSON.parse(f)
end

start_pos = Point(episode_dict["start_pos"][1], episode_dict["start_pos"][2])
goal_pos = Point(episode_dict["goal_pos"][1], episode_dict["goal_pos"][2])

# Get episode length
num_epochs = episode_dict["num_epochs"]-1


# Define Mersenne Twister for reproducibility
rng = MersenneTwister(2)

# Create dynamics model and MDP
uav_dynamics = MultiRotorUAVDynamicsModel(MDP_TIMESTEP, ACC_NOISE_STD)
pc_hopon_mdp = ControlledMultiRotorHopOnMDP(uav_dynamics)
uc_hopoff_mdp = HopOffMDP()

# Create drone
drone = Drone()

# Create SDMC Simulator and get initial epoch
sdmc_sim = SDMCSimulator(episode_dict["epochs"], uav_dynamics, start_pos, goal_pos, rng)

# Load policies
hopon_policy = load(hopon_file,"policy")
hopoff_policy = load(hopoff_file,"policy")

# Create graph solution
graph_planner = GraphSolution(drone)
setup_graph(graph_planner, start_pos, goal_pos, get_epoch0_dict(sdmc_sim))

# Setup weight function
flight_edge_wt_fn(u,v) = flight_edge_cost_valuefn(uav_dynamics, hopon_policy, u, v)

