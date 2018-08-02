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
function flight_edge_cost_nominal(u::CarDroneVertex, v::CarDroneVertex, d::Drone)
    dist::Float64 = point_dist(u.pos, v.pos)
    cost::Float64 = FLIGHT_COEFFICIENT*dist
    if v.time_stamp < Inf
        cost += TIME_COEFFICIENT*(v.time_stamp - u.time_stamp)
    else
        if u.time_stamp == 0.0
            cost = Inf
        else
            # TODO - Is this right????
            cost += TIME_COEFFICIENT*(dist*2.0/d.max_speed)
        end
    end
    # println("Nominal edge - ",u.idx," to ",v.idx," of cost ",cost)
    return cost
end


# ASSUME this is inside distance limits
function flight_edge_cost_valuefn(udm::UDM, hopon_policy::PartialControlHopOnOffPolicy,
                                  flight_policy::LocalApproximationValueIterationPolicy,
                                  u::CarDroneVertex, v::CarDroneVertex, d::Drone) where {UDM <: UAVDynamicsModel}

    rel_pos = Point(u.pos.x - v.pos.x, u.pos.y - v.pos.y)
    rel_state = get_state_at_rest(udm, rel_pos)

    cost = 0.0

    if v.time_stamp == Inf
        cost += -value(flight_policy, rel_state) + FLIGHT_REACH_REWARD
    else
        horizon = convert(Int,round((v.time_stamp-u.time_stamp)/MDP_TIMESTEP))
        if horizon <= HORIZON_LIM
            hopon_state = ControlledHopOnStateAugmented(rel_state,horizon)
            cost += -value(hopon_policy.in_horizon_policy,hopon_state)
            #end
        else
            # Use outhorizon cost but also additional cost for time and/or distance
            addtn_time_cost = TIME_COEFFICIENT*MDP_TIMESTEP*(horizon - HORIZON_LIM)
            hopon_outhor_state = ControlledHopOnStateAugmented(rel_state,HORIZON_LIM)
            cost += -value(hopon_policy.in_horizon_policy, hopon_outhor_state) + addtn_time_cost
        end
    end

    if u.is_car && v.is_car && u.car_id != v.car_id
        #cost = 0.5*cost
        #println("Car-Car VFn edge - ",u.idx," to ",v.idx," of cost ",cost)
    end

    return cost
end