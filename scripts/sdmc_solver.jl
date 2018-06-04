# mutable struct FullSDMCState
#     drone::Drone
#     sdmc_state::SDMCState 
#     ... # to be continued
# end

# NOTE - If the macro action is at a stage outside the bounds of the drone, just fly on the st.line path
# towards the goal till it is within range

# Read in the episode filename from args
ep_file = ARGS[1]

open(ep_file,"r") do f
    episode_dict = JSON.parse(f)
end


# Define Mersenne Twister for reproducibility
rng = MersenneTwister(2)

# Create dynamics model and MDP
uav_dynamics = MultiRotorUAVDynamicsModel(MDP_TIMESTEP, ACC_NOISE_STD)
pc_hopon_mdp = ControlledMultiRotorHopOnMDP(uav_dynamics)

# Create drone


# Create SDMC Simulator



# Load policies



# Create graph solution