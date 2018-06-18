mutable struct SimpleVListGraph{V} <: AbstractGraph{V,Edge{V}}
    vertices::Vector{V}
end

# function SimpleVListGraph(verts::Vector{V}) where V
#     return SimpleVListGraph(verts)
# end

function Graphs.add_vertex!(g::SimpleVListGraph{V}, v::V) where V
    push!(g.vertices, v)
end

function Graphs.num_vertices(g::SimpleVListGraph{V}) where V
    return length(g.vertices)
end


# CarDroneVertex for drone flight waypoints
# Will typically just be the start and goal
mutable struct CarDroneVertex
    idx::Int
    pos::Point
    time_stamp::Float64
    last_time_stamp::Float64
    is_car::Bool
    car_id::String
end

CarDroneVertex() = CarDroneVertex(0,Point(),0.0,0.0,false)

function CarDroneVertex(idx::Int, pos::Point, time_stamp::Float64, is_car::Bool, car_id::String="")
    return CarDroneVertex(idx, pos, time_stamp, time_stamp, is_car, car_id)
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
    max_axis_dist = max(abs(u.pos.x - v.pos.x),abs(u.pos.y - v.pos.y))
    return XYDOT_LIM*(v.time_stamp - u.time_stamp) > max_axis_dist*VALID_FLIGHT_EDGE_DIST_RATIO
end

function coast_edge_cost(u::CarDroneVertex, v::CarDroneVertex)
    return TIME_COEFFICIENT*(v.time_stamp - u.time_stamp)
end

# Nominal flight edge cost when pc policy not used
# TODO : Need a model of flight cost for such flight edges
function flight_edge_cost_nominal(u::CarDroneVertex, v::CarDroneVertex, d::Drone)
    dist::Float64 = point_dist(u.pos, v.pos)
    cost::Float64 = FLIGHT_COEFFICIENT*dist
    if v.time_stamp < Inf
        cost += TIME_COEFFICIENT*(v.time_stamp - u.time_stamp)
    else
        cost += TIME_COEFFICIENT*(dist/d.max_speed)
    end
    # println("Nominal edge - ",u.idx," to ",v.idx," of cost ",cost)
    return cost
end


# ASSUME this is inside distance limits
function flight_edge_cost_valuefn(udm::UDM, hopon_policy::PartialControlHopOnOffPolicy,
         u::CarDroneVertex, v::CarDroneVertex, d::Drone) where {UDM <: UAVDynamicsModel}
    horizon = convert(Int,round((v.time_stamp-u.time_stamp)/MDP_TIMESTEP))

    rel_pos = Point(u.pos.x - v.pos.x, u.pos.y - v.pos.y)
    rel_state = get_state_at_rest(udm, rel_pos)

    if horizon > HORIZON_LIM+2 && (abs(rel_state.x) > 1.05*XY_LIM || abs(rel_state.y) > 1.05*XY_LIM)
        return flight_edge_cost_nominal(u,v,d)
    end

    # Initialize with additional distance cost, if any
    cost = FLIGHT_COEFFICIENT*max(0,point_norm(rel_pos) - XY_LIM*1.05*sqrt(2))

    if horizon < HORIZON_LIM
        hopon_state = ControlledHopOnStateAugmented(rel_state,false,horizon)
        cost += -value(hopon_policy.in_horizon_policy,hopon_state)
    else
        # Use outhorizon cost but also additional cost for time and/or distance
        addtn_time_cost = TIME_COEFFICIENT*(horizon - HORIZON_LIM)
        hopon_outhor_state = ControlledHopOnStateAugmented(rel_state,false,0.)
        cost += -value(hopon_policy.out_horizon_policy, hopon_outhor_state) + addtn_time_cost
    end

    cost = cost + HOP_REWARD

    #println("Valuefn edge - ",u.idx," to ",v.idx," of cost ",cost)

    return cost
end