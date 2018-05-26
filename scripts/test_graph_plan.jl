# This script just tests if the GraphSolution object in graph_plan
# parses and updates the simulator correctly with each epoch
using JSON
using HitchhikingDrones

EPISODE_FILE = "../data/example_1ep.json"

# Assume drone params obtained from somewhere
drone = Drone(1)

graph_soln = GraphSolution(drone)

episode_dict = Dict()

open(EPISODE_FILE,"r") do f
    global episode_dict
    episode_dict = JSON.parse(readstring(f))
end

start_pos = Point(episode_dict["start_pos"][1], episode_dict["start_pos"][2])
goal_pos = Point(episode_dict["goal_pos"][1], episode_dict["goal_pos"][2])
num_epochs = episode_dict["num_epochs"]


epochs = episode_dict["epochs"]
setup_graph(graph_soln, start_pos, goal_pos, epochs["1"])