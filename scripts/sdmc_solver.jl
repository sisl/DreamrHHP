mutable struct FullSDMCState
    drone::Drone
    sdmc_state::SDMCState 
    ... # to be continued
end

# IMP - If the macro action is at a stage outside the bounds of the drone, just fly on the st.line path
# towards the goal till it is within range