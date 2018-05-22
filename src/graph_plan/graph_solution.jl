# This is a general graph solution object that currently could use either A*
# or D*-Lite to replan. It has methods for updating all car route vertices
# and replanning from a certain vertex in the graph.
# At the beginning, drone vertices are only the initial time start and the
# time-unconstrained goal vertex but other drone vertices may be added at an
# intermediate stage, especially when lower level policies are aborted

# TODO : Use IncidenceList or Graph?
mutable struct GraphSolution
    car_map::Dict{String,Car}
    drone::Drone
    max_car_speed::Float64
    goal_vertex::Vertex
    car_drone_graph::Graph{Vertex,ExEdge{Vertex}}
end

function GraphSolution(_drone::Drone, _max_car_speed::Float64)
    # define car_drone_graph

end

function setupGraph(startpos,carmap)


    # Setup car_map for first epoch

end


function update_car_with_epoch()

    # for non-first epoch
end
