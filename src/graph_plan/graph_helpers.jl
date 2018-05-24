mutable struct SimpleVListGraph{V} <: AbstractGraph{V}
    vertices::Vector{V}
end

function Graphs.add_vertex!(g::SimpleVListGraph{V}, v::V) where V
    push!(vertices, v)
end


# CarDroneVertex for drone flight waypoints
# Will typically just be the start and goal
mutable struct CarDroneVertex
    idx::Int
    pos::Point
    time_stamp::Float64
    last_time_stamp::Float64
    is_car::Bool
end

CarDroneVertex() = CarDroneVertex(0,Point(),0.0,0.0,false)

function CarDroneVertex(idx::Int, pos::Point, time_stamp::Float64, is_car::Bool)
    return CarDroneVertex(idx, pos, time_stamp, time_stamp, is_car)
end


# For new vertex, last updated TS is same
# function CarDroneVertex(idx::Int, pos::Point, time_stamp::Float64)
#     return CarDroneVertex(idx, pos, time_stamp, time_stamp)
# end

function Graphs.vertex_index(v::CarDroneVertex, g::SimpleVListGraph{CarDroneVertex})
    return v.idx
end

# Can drone even make the flight edge in time
function is_valid_flight_edge(u::CarDroneVertex, v::CarDroneVertex, d::Drone)
    return point_dist(u.pos, v.pos) > (VALID_FLIGHT_EDGE_DIST_RATIO*d.max_speed*(v.time_stamp - u.time_stamp))
end

function coast_edge_cost(u::CarDroneVertex, v::CarDroneVertex)
    return TIME_COEFFICIENT*(v.time_stamp - u.time_stamp)
end

# Nominal flight edge cost when pc policy not used
# TODO : Need a model of flight cost for such flight edges
function flight_edge_cost_nominal(u::CarDroneVertex, v::CarDroneVertex, d::Drone)
    dist::Float64 = point_dist(u.pos, v.pos)
    cost::Float64 = FLIGHT_COEFFICIENT*dist
    if v.time_stamp < Inf:
        # This is unlikely for now
        cost += TIME_COEFFICIENT*(v.time_stamp - u.time_stamp)
    else
        cost += TIME_COEFFICIENT*(dist/d.max_speed)
    end
end