# Vertex for drone flight waypoints
# Will typically just be the start and goal
mutable struct Vertex
    idx::Int
    pos::Point
    time_stamp::Float64
end

Vertex() = Vertex(0,Point(),0.0)

# Vertex type for car route waypoints
# Will be updated often
# mutable struct CarRouteVertex <: Vertex
#     idx::Int
#     pos::Point
#     time_stamp::Float64 # Keeping separate because this will be changed often
#     car_idx::String
# end

# CarRouteVertex() = CarRouteVertex(0,Point(),0.0,"null")

# Vertex = Union{DroneVertex, CarRouteVertex}

function Graphs.vertex_index(v::Vertex, g::IncidenceList{Vertex,ExEdge{Vertex}})
    return v.idx
end

# Can drone even make the flight edge in time
function is_valid_flight_edge(u::Vertex, v::Vertex, d::Drone)
    return point_dist(u.pos, v.pos) > (VALID_FLIGHT_EDGE_DIST_RATIO*d.max_speed*(v.time_stamp - u.time_stamp))
end

function coast_edge_cost(u::Vertex, v::Vertex)
    return TIME_COEFFICIENT*(v.time_stamp - u.time_stamp)
end

# Nominal flight edge cost when pc policy not used
# TODO : Need a model of flight cost for such flight edges
function flight_edge_cost_nominal(u::Vertex, v::Vertex, d::Drone)
    dist::Float64 = point_dist(u.pos, v.pos)
    cost::Float64 = FLIGHT_COEFFICIENT*dist
    if v.time_stamp < Inf:
        # This is unlikely for now
        cost += TIME_COEFFICIENT*(v.time_stamp - u.time_stamp)
    else
        cost += TIME_COEFFICIENT*(dist/d.max_speed)
    end
end

# For edges where drone must fly to a car
# This is computed using the hopon policy with Certainty Principle
# NOTE - This is just a wrapper - the state needs to be created from the vertices
function flight_edge_cost_valuefn(policy::LocalApproximationValueIterationPolicy, s::S) where S
    return policy.value(s)
end