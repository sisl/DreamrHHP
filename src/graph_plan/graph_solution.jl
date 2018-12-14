# This is a general graph solution object
# It has methods for updating all car route vertices
# and replanning from a certain vertex in the graph.
# At the beginning, drone vertices are only the initial time start and the
# time-unconstrained goal vertex but other drone vertices may be added at an
# intermediate stage, especially when lower level policies are aborted
# TODO : Check that ALL MEMBERS are useful
"""
Data structure that tracks and updates solutions from the higher level graph planner.
"""
mutable struct GraphSolution
    car_map::Dict{String,Car}
    drone::Drone
    max_car_speed::Float64
    goal_idx::Int
    next_start_idx::Int
    has_next_start::Bool
    n_vertices::Int
    curr_time::Float64
    car_drone_graph::SimpleVListGraph{CarDroneVertex}
    route_vert_id_to_idx::Dict{String,Int}
    drone_vertex_idxs::Vector{Int}
    flight_edge_wts::Dict{Tuple{Int,Int},Float64}
    curr_soln_idx_path::Vector{Int}
    curr_best_soln_value::Float64
    has_next_macro_action::Bool
    future_macro_actions_values::Vector{Tuple{Tuple{CarDroneVertex,CarDroneVertex}, Float64}}
    per_plan_considered_cars::Dict{Int,Set{String}}
    params::Parameters
end

function GraphSolution(drone::Drone, params::Parameters)
    # define car_drone_graph
    car_drone_graph = SimpleVListGraph(Vector{CarDroneVertex}(undef,0))

    return GraphSolution(Dict{String,Car}(), drone, 
        params.scale_params.MAX_CAR_SPEED, 0, 0, false, 0, 0.0, car_drone_graph, 
        Dict{String,Int}(), Vector{Int}(undef,0), Dict{Tuple{Int,Int},Float64}(),
        Vector{Int}(undef,0), Inf, false, Vector{Tuple{Tuple{CarDroneVertex,CarDroneVertex}, Float64}}(undef,0),
        Dict{Int,Set{String}}(), params)
end

"""
    setup_graph(gs::GraphSolution, start_pos::Point, goal_pos::Point, epoch0::Dict, start_time::Float64=0.0, goal_time::Float64=Inf)

Initialize graph for first epoch. Assume that updates to successive epochs are only in car_routes.
"""
function setup_graph(gs::GraphSolution, start_pos::Point, goal_pos::Point, epoch0::Dict, start_time::Float64=0.0, goal_time::Float64=Inf)

    # Set current time
    gs.curr_time = convert(Float64, epoch0["time"])

    # Initialize start vertex
    gs.n_vertices += 1
    add_vertex!(gs.car_drone_graph, CarDroneVertex(gs.n_vertices, start_pos, start_time,false))
    gs.next_start_idx = gs.n_vertices
    gs.has_next_start = true
    push!(gs.drone_vertex_idxs, gs.n_vertices)

    # Initialize goal vertex
    gs.n_vertices += 1
    add_vertex!(gs.car_drone_graph, CarDroneVertex(gs.n_vertices, goal_pos, goal_time,false))
    gs.goal_idx = gs.n_vertices
    push!(gs.drone_vertex_idxs, gs.n_vertices)

    # Now add vertices for car route
    # NOTE - Ordering of cars is immaterial here
    epoch_cars = epoch0["car-info"]
    for (car_id, car_info) in epoch_cars

        route_info = car_info["route"]

        # First add vertices if there are any
        if route_info != nothing
            # Next route vertex is the first
            first_route_idx = gs.n_vertices+1

            # NOTE - parse here done for floating points so that intermediate points can be inserted later
            for (id, timept) in sort(collect(route_info),by=x->parse(Float64, x[1]))
                gs.n_vertices += 1
                add_vertex!(gs.car_drone_graph, CarDroneVertex(gs.n_vertices, Point(timept[1][1], timept[1][2]), timept[2], true, car_id))
                gs.route_vert_id_to_idx[string(car_id,"-",id)] = gs.n_vertices
            end

            last_route_idx = gs.n_vertices

            # Now add appropriate car object
            gs.car_map[car_id] = Car([first_route_idx, last_route_idx])
            # info("Car ",car_id," has been added!")
        else
            # Add inactive car - this should not happen at first epoch though
            @warn "Inactive Car ",car_id," in first epoch!"
            gs.car_map[car_id] = InactiveCar()
        end
    end
end


"""
    update_cars_with_epoch(gs::GraphSolution, epoch::Dict)

Given the information of the next epoch, perform necessary updates to the GraphSolution object. 
"""
function update_cars_with_epoch(gs::GraphSolution, epoch::Dict)

    # Set current time
    gs.curr_time = convert(Float64, epoch["time"])

    # Iterate over cars, add if new or update if old
    epoch_cars = epoch["car-info"]

    for (car_id, car_info) in epoch_cars

        # Update car route if it exists
        if haskey(gs.car_map, car_id)

            this_car = gs.car_map[car_id]
            route_info = car_info["route"]

            if route_info != nothing

                # Update times of all future vertices
                # NOTE - This assumes no new vertices added (for now)
                sorted_route = sort(collect(route_info), by=x->parse(Float64, x[1]))
                for (id, timept) in sorted_route
                    # TODO: Check for equality of points?
                    prev_time = gs.car_drone_graph.vertices[gs.route_vert_id_to_idx[string(car_id,"-",id)]].time_stamp
                    gs.car_drone_graph.vertices[gs.route_vert_id_to_idx[string(car_id,"-",id)]].last_time_stamp = prev_time
                    gs.car_drone_graph.vertices[gs.route_vert_id_to_idx[string(car_id,"-",id)]].time_stamp = timept[2]
                end

                # Check if the next waypoint of route has been updated
                first_route_idx = gs.route_vert_id_to_idx[string(car_id,"-",sorted_route[1][1])]
                if this_car.route_idx_range[1] != first_route_idx
                    # info("Car ",car_id," has updated its next route point to ",first_route_idx)
                    this_car.route_idx_range[1] = first_route_idx
                end
            else
                this_car.active = false
            end
        else
            # New car - add as above
            route_info = car_info["route"]

            if route_info != nothing
                first_route_idx = gs.n_vertices+1

                for (id, timept) in sort(collect(route_info),by=x->parse(Float64, x[1]))
                    gs.n_vertices += 1
                    add_vertex!(gs.car_drone_graph, CarDroneVertex(gs.n_vertices, Point(timept[1][1], timept[1][2]), timept[2], true, car_id))
                    gs.route_vert_id_to_idx[string(car_id,"-",id)] = gs.n_vertices
                end

                last_route_idx = gs.n_vertices

                gs.car_map[car_id] = Car([first_route_idx, last_route_idx])
                # info("Car ",car_id," has been added!")
            end
        end
    end
end

# Add a new vertex and return index of it
"""
    add_drone_vertex(gs::GraphSolution, pos::Point, time_stamp::Float64)

Insert a new drone vertex to the graph at the given position and current time stamp.
"""
function add_drone_vertex(gs::GraphSolution, pos::Point, time_stamp::Float64)
    gs.n_vertices += 1
    add_vertex!(gs.car_drone_graph, CarDroneVertex(gs.n_vertices, pos, time_stamp, false))
    push!(gs.drone_vertex_idxs,gs.n_vertices)
    return gs.n_vertices
end

"""
    remove_last_drone_vertex(gs::GraphSolution)

Remove the last vertex from the graph, assuming it is a drone vertex.
"""
function remove_last_drone_vertex(gs::GraphSolution)
    gs.n_vertices -= 1
    remove_last_vertex!(gs.car_drone_graph)
    @assert pop!(gs.drone_vertex_idxs) != gs.goal_idx
end

"""
    add_new_start(gs::GraphSolution, pos::Point, time_stamp::Float64)

Insert a new source vertex into the graph which represents the current drone state.
"""
function add_new_start(gs::GraphSolution, pos::Point, time_stamp::Float64)
    gs.next_start_idx = add_drone_vertex(gs, pos, time_stamp)
    gs.has_next_start = true
end

"""
    revert_new_start(gs::GraphSolution, idx::Int)

Negate the potential new start vertex for the graph.
"""
function revert_new_start(gs::GraphSolution, idx::Int)
    remove_last_drone_vertex(gs)
    gs.next_start_idx = idx
    gs.has_next_start = true
end

"""
    astar_heuristic(gs::GraphSolution, v::CarDroneVertex)

Compute the (very loose) admissible heuristic for the higher level graph.
"""
function astar_heuristic(gs::GraphSolution, v::CarDroneVertex)
    return gs.params.cost_params.TIME_COEFFICIENT*point_dist(v.pos, 
            gs.car_drone_graph.vertices[gs.goal_idx].pos)/gs.max_car_speed
end


"""
    edge_weight_function_recompute(flight_edge_wt_fn::Function, gs::GraphSolution, 
                                        u::CarDroneVertex, v::CarDroneVertex)

Recompute the edge weight for the edge in the higher level graph, based on the kind of edge
"""
function edge_weight_function_recompute(flight_edge_wt_fn::Function, gs::GraphSolution, 
                                        u::CarDroneVertex, v::CarDroneVertex)
    if u.is_car && v.is_car && u.car_id == v.car_id
        # Coasting edge
        return coast_edge_cost(u,v)
    elseif !v.is_car
        # Flight edge to drone vertex 
        return flight_edge_cost_nominal(u,v,gs.drone)
    else
        # Constrained flight edge
        return flight_edge_wt_fn(u,v)
    end
end


"""
    edge_weight_function_lookup(flight_edge_wt_fn::Function, gs::GraphSolution, u::CarDroneVertex, v::CarDroneVertex)

Lookup the edge weight function from the cache instead of recomputing from scratch.
"""
function edge_weight_function_lookup(flight_edge_wt_fn::Function, gs::GraphSolution, u::CarDroneVertex, v::CarDroneVertex)
    if u.is_car && v.is_car && u.car_id == v.car_id  
        edge_weight_val = coast_edge_cost(u,v)
    else
        # Flight edge from value function - either constrained or unconstrained
        edge_weight_val = get(gs.flight_edge_wts, (u,v), Inf)
        if edge_weight_val == Inf
            # Not present - compute weight and update
            edge_weight_val = flight_edge_wt_fn(u,v)
            gs.flight_edge_wts[(u.idx,v.idx)] = edge_weight_val
        else
            # If either vertex time has changed significantly, update both and recompute
            if abs(u.time_stamp - u.last_time_stamp) > gs.params.time_params.WAYPT_TIME_CHANGE_THRESHOLD || 
                abs(v.time_stamp - v.last_time_stamp) > gs.params.time_params.WAYPT_TIME_CHANGE_THRESHOLD
                u.last_time_stamp = u.time_stamp
                v.last_time_stamp = v.time_stamp
                edge_weight_val = flight_edge_wt_fn(u,v)
                gs.flight_edge_wts[(u.idx,v.idx)] = edge_weight_val
            end
        end
    end

    # Return the appropriate value
    return edge_weight_val
end

# If no new waypoints for car routes, iterate over car_routes using route_idx_range
"""
    car_route_idx_list_simple(gs::GraphSolution, car_id::String)

Obtain the list of integer indices of the car route waypoints for a car. In the simple case, with no future insertions of 
intermediate waypoints, this can just return the subarray based on route_idx_range.
"""
function car_route_idx_list_simple(gs::GraphSolution, car_id::String)
    return Vector{Int}(gs.car_map[car_id].route_idx_range[1] : gs.car_map[car_id].route_idx_range[2])
end

# function car_route_idx_list_general(gs::GraphSolution, car_id::String)
#     # DO sorted_route with the current car route
#     # Return the vector of graph vertex IDs
# end

"""
    update_next_start(gs::GraphSolution, next_start_idx::Int)

Trivial - just update the next start index of the higher level graph.
"""
function update_next_start(gs::GraphSolution, next_start_idx::Int)
    gs.next_start_idx = next_start_idx
    gs.has_next_start = true
end


# Whatever the next replan start vertex is, plan from it towards goal
# NOTE - Updating the next start will be done by higher layer
"""
    plan_from_next_start(gs::GraphSolution, flight_edge_wt_fn::Function, valid_edge_fn::Function)

Compute the current best sequences of macro actions and corresponding values from the current start vertex.
"""
function plan_from_next_start(gs::GraphSolution, flight_edge_wt_fn::Function, valid_edge_fn::Function)

    # Set up heuristic and edge_weight_functions
    # TODO : What's the right way to just do this once?
    heuristic(v) = astar_heuristic(gs, v)
    # heuristic(v) = 0.0
    edge_wt_fn(u,v) = edge_weight_function_lookup(flight_edge_wt_fn, gs, u, v)

    # println("Planning from - ",gs.car_drone_graph.vertices[gs.next_start_idx])

    # Reset the considered cars dict for each vertex
    # Add the next start with empty set (so its successor can copy from it)
    gs.per_plan_considered_cars = Dict{Int, Set{String}}()
    gs.per_plan_considered_cars[gs.next_start_idx] = Set{String}()

    astar_path_soln = astar_light_shortest_path_implicit(gs.car_drone_graph,edge_wt_fn,gs.next_start_idx,
                                                   GoalVisitorImplicit(gs, valid_edge_fn),heuristic)

    # Obtain path and its current cost
    # This is the best path regardless. DON'T NEED TO RECOMPUTE OLD PATH
    if astar_path_soln.dists[gs.goal_idx] == Inf
        @warn "Goal vertex is currently unreachable. No next macro action!"
        gs.has_next_macro_action = false
        return
    end

    # Goal is reachable
    gs.has_next_macro_action = true

    # Extract path from next start to goal
    # First clear out current solution
    empty!(gs.curr_soln_idx_path)

    # Walk back path to current start
    unshift!(gs.curr_soln_idx_path, gs.goal_idx)
    curr_vertex_idx = gs.goal_idx

    while curr_vertex_idx != gs.next_start_idx
        prev_vertex_idx = astar_path_soln.parent_indices[curr_vertex_idx]
        unshift!(gs.curr_soln_idx_path, prev_vertex_idx)
        curr_vertex_idx = prev_vertex_idx
    end


    gs.curr_best_soln_value = astar_path_soln.dists[gs.goal_idx]

    # println("Curr soln idx path:")
    # for idx in gs.curr_soln_idx_path
    #     println("Idx - ",idx," ; cost - ",astar_path_soln.dists[idx])
    # end


    # now update the future macro_actions
    empty!(gs.future_macro_actions_values)

    last_idx = gs.curr_soln_idx_path[1]
    for (i,idx) in enumerate(gs.curr_soln_idx_path[2:end-1])
        # Compute the value function cost of the edge
        macro_action_val = astar_path_soln.dists[idx] - astar_path_soln.dists[last_idx]
        if gs.car_drone_graph.vertices[idx].is_car != gs.car_drone_graph.vertices[last_idx].is_car
            # Either car to fin or start to car - update and set macro action
            push!(gs.future_macro_actions_values,
                ((gs.car_drone_graph.vertices[last_idx], gs.car_drone_graph.vertices[idx]), macro_action_val))
            last_idx = idx
        else
            # NOTE - Will never be flight to flight; So CAR to CAR
            if gs.car_drone_graph.vertices[idx].is_car == false && gs.car_drone_graph.vertices[last_idx].is_car
                @warn "Flight to flight edge macro action intermediate?"
            else
                if gs.car_drone_graph.vertices[idx].car_id != gs.car_drone_graph.vertices[last_idx].car_id
                    # Car to new car
                    push!(gs.future_macro_actions_values,
                        ((gs.car_drone_graph.vertices[last_idx], gs.car_drone_graph.vertices[idx]), macro_action_val))
                    last_idx = idx
                else
                    # Add coast edge if next vertex is diff car or flight
                    # println("In coast edge loop")
                    # println(gs.car_drone_graph.vertices[gs.curr_soln_idx_path[i+2]])
                    if (gs.car_drone_graph.vertices[gs.curr_soln_idx_path[i+2]].car_id != gs.car_drone_graph.vertices[idx].car_id) || 
                        gs.car_drone_graph.vertices[gs.curr_soln_idx_path[i+2]].is_car == false
                        push!(gs.future_macro_actions_values,
                        ((gs.car_drone_graph.vertices[last_idx], gs.car_drone_graph.vertices[idx]), macro_action_val))
                        last_idx = idx
                    end
                end
            end
        end
    end


    # Append the last one regardless
    last_val = gs.curr_best_soln_value - astar_path_soln.dists[last_idx]
    push!(gs.future_macro_actions_values, ((gs.car_drone_graph.vertices[last_idx], gs.car_drone_graph.vertices[gs.goal_idx]), last_val))
    
    # Debug : Display (macro action, value)
    # for mav in gs.future_macro_actions_values
    #     println(mav)
    # end
end


# Returns a vector of pairs of vertices in order that represent the subsequent macro actions 
# currently planned by the agent
# NOTE - Just do this directly at top level
function get_future_macro_actions_values(gs::GraphSolution)

    if gs.has_next_macro_action == false
        warn("Current replan has no feasible solution. Use previous plan or abort")
        return nothing
    end

    return gs.future_macro_actions_values
end


mutable struct GoalVisitorImplicit <: AbstractDijkstraVisitor
    gs::GraphSolution
    valid_edge_fn::Function
end


function Graphs.include_vertex!(vis::GoalVisitorImplicit, u::CarDroneVertex, v::CarDroneVertex, d::Float64, nbrs::Vector{Int})

    # NOTE - nbrs is updated in place!
    # println(v," is popped")

    if v.idx == vis.gs.goal_idx
        # readline()
        return false
    end

    # Now add flight edges to all other applicable car and drone verts
    # Assume predecessor had some passedCars or empty set
    vis.gs.per_plan_considered_cars[v.idx] = copy(vis.gs.per_plan_considered_cars[u.idx])
    # vis.gs.per_plan_considered_cars[v.idx] = Set{String}()

    # If pred was a car vertex and of a different idx, update passed cars
    if u.is_car == true && u.car_id != v.car_id
        push!(vis.gs.per_plan_considered_cars[v.idx], u.car_id)
    end

    # First do special work if it is a car route vertex
    # Add the next route vertex (and no others) from that car
    if v.is_car == true
        self_route_idxs = car_route_idx_list_simple(vis.gs, v.car_id)

        # Find the position of the current index on the list
        v_pos = findfirst(self_route_idxs.==v.idx)

        # If it is NOT the last element of the list, add the next route vert to nbrs
        if v_pos < length(self_route_idxs)
            push!(nbrs,self_route_idxs[v_pos+1])
        end

        # println("Neighbors:")
        # for n in nbrs
        #     print(n,", ")
        # end
        # NOTE : If this is the first coast vertex, don't do any more
        # Must do at least 1 coast edge
        if u.is_car == false || u.car_id != v.car_id || (u.idx == v.idx && v.idx != vis.gs.car_map[v.car_id].route_idx_range[2])
            return true
        end
    end


    # Iterate over cars that are not the same and not in passed cars
    for car_id in collect(keys(vis.gs.car_map))
        if car_id == v.car_id || car_id in vis.gs.per_plan_considered_cars[v.idx] || vis.gs.car_map[car_id].active==false
            continue
        end

        # println("Considering ",car_id, " from ",v.car_id)

        # Iterate over car route vertices
        car_route_idxs= car_route_idx_list_simple(vis.gs, car_id)

        # NOTE - This assumes that all vertices are later in time
        # Epoch update function has handled this
        for next_idx in car_route_idxs
            if vis.valid_edge_fn(v,vis.gs.car_drone_graph.vertices[next_idx],vis.gs.drone)
                push!(nbrs,next_idx)
            end
        end
    end

    # Iterate over remaining drone vertices and add flight edges to them
    # TODO - Fix this later when you may have interm drone
    for dvtx_idx in vis.gs.drone_vertex_idxs
        if dvtx_idx != v.idx && vis.valid_edge_fn(v,vis.gs.car_drone_graph.vertices[dvtx_idx],vis.gs.drone)
            push!(nbrs, dvtx_idx)
        end
    end

    # println("Neighbors:")
    # for n in nbrs
    #     print(n,", ")
    # end

    # readline()

    return true
end
