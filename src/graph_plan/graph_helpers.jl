"""
Custom graph structure which only has a list of vertices; edges generated implicitly.
"""
mutable struct SimpleVListGraph{V} <: AbstractGraph{V,Edge{V}}
    vertices::Vector{V}
end

function Graphs.add_vertex!(g::SimpleVListGraph{V}, v::V) where V
    push!(g.vertices, v)
end

function Graphs.num_vertices(g::SimpleVListGraph{V}) where V
    return length(g.vertices)
end

# Non Graphs function
function remove_last_vertex!(g::SimpleVListGraph{V}) where V
    pop!(g.vertices)
end


"""
Data structure representing a vertex of the layer 1 graph. Has all the information to represent either a
car's waypoint or the drone's current position.
"""
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


Graphs.vertex_index(v::CarDroneVertex, g::SimpleVListGraph{CarDroneVertex}) = v.idx

"""
    is_valid_flight_edge(u::CarDroneVertex, v::CarDroneVertex, d::Drone, params::Parameters)

Nominal check for whether the distance between vertices is less than a maximum fraction of the
maximum possible straight line distance the agent can traverse in the remaining time.
"""
function is_valid_flight_edge(u::CarDroneVertex, v::CarDroneVertex, d::Drone, params::Parameters)
    return d.max_speed*(v.time_stamp - u.time_stamp) > 
           point_dist(u.pos, v.pos)*params.scale_params.VALID_FLIGHT_EDGE_DIST_RATIO
end

function coast_edge_cost(u::CarDroneVertex, v::CarDroneVertex, params::Parameters)
    return params.cost_params.TIME_COEFFICIENT*(v.time_stamp - u.time_stamp)
end

"""
    flight_edge_cost_nominal(u::CarDroneVertex, v::CarDroneVertex, udm::UDM, 
                             d::Drone, params::Parameters, energy_time_alpha::Float64=0.5) where {UDM <: UAVDynamicsModel

Compute the cost for nominally traversing a flight edge. If time constrained, use the time difference and the distance cost.
If time unconstrained, use time from bang-bang strategy.
"""
function flight_edge_cost_nominal(u::CarDroneVertex, v::CarDroneVertex, udm::UDM, 
                                  params::Parameters, energy_time_alpha::Float64=0.5) where {UDM <: UAVDynamicsModel}
    dist = point_dist(u.pos, v.pos)
    if v.time_stamp < Inf
        cost = (1.0 - energy_time_alpha)*params.cost_params.FLIGHT_COEFFICIENT*dist +
               energy_time_alpha*params.cost_params.TIME_COEFFICIENT*(v.time_stamp - u.time_stamp)
    else
        rel_pos = Point(u.pos.x - v.pos.x, u.pos.y - v.pos.y)
        temp_start_state = get_state_at_rest(udm, rel_pos)
        (rew, tval) = bang_bang_reward_time(dist, temp_start_state, params)
        cost = -rew
    end
    return cost
end


"""
    flight_edge_cost_valuefn(udm::UDM, hopon_policy::PartialControlHopOnOffPolicy,
                                  flight_policy::LocalApproximationValueIterationPolicy,
                                  u::CarDroneVertex, v::CarDroneVertex, params::Parameters) where {UDM <: UAVDynamicsModel}

Compute the cost for traversing a flight edge according to the value function. If time constrained, use the hopon policy 
value function, or else use the unconstrained flight value function.
"""
function flight_edge_cost_valuefn(udm::UDM, hopon_policy::PartialControlHopOnOffPolicy,
                                  flight_policy::LocalApproximationValueIterationPolicy,
                                  u::CarDroneVertex, v::CarDroneVertex, params::Parameters) where {UDM <: UAVDynamicsModel}

    rel_pos = Point(u.pos.x - v.pos.x, u.pos.y - v.pos.y)
    rel_state = get_state_at_rest(udm, rel_pos)

    cost = 0.0

    if v.time_stamp == Inf
        cost += -value(flight_policy, rel_state) + params.cost_params.FLIGHT_REACH_REWARD
    else
        horizon = convert(Int,round((v.time_stamp-u.time_stamp)/params.time_params.MDP_TIMESTEP))
        if horizon <= params.time_params.HORIZON_LIM
            hopon_state = ControlledHopOnStateAugmented(rel_state,horizon)
            cost += -value(hopon_policy.in_horizon_policy,hopon_state)
            #end
        else
            # Use outhorizon cost but also additional cost for time and/or distance
            addtn_time_cost = params.cost_params.TIME_COEFFICIENT*params.time_params.MDP_TIMESTEP*(horizon - params.time_params.HORIZON_LIM)
            hopon_outhor_state = ControlledHopOnStateAugmented(rel_state,params.time_params.HORIZON_LIM)
            cost += -value(hopon_policy.in_horizon_policy, hopon_outhor_state) + addtn_time_cost
        end
    end

    return cost
end