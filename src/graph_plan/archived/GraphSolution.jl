using Graphs
using DataStructures
using IJulia
include("Types.jl")
include("Solution.jl")
include("GraphHelpers.jl")
include("AStarVisitor.jl")



# CURRENTLY ASSUME SINGLE DRONE
mutable struct GraphSolution <: Solution

  # Member variables
  listOfCars::Vector{Car}
  drone::Drone
  maxCarSpeed::Float64
  maxSystemSpeed::Float64
  nCars::Int
  nVertices::Int
  nEdges::Int
  startVertex::Vertex
  goalVertex::Vertex
  currSolnPath::Vector{Vertex}
  currSolnCost::Float64
  problemGraph::IncidenceList{Vertex,ExEdge{Vertex}}
  carIdxRange::Vector{Tuple{Int,Int}}
  firstRouteIdxAltered::Vector{Int}


  function GraphSolution(_listOfCars::Vector{Car},
                         _drone::Drone,
                         _maxCarSpeed::Float64)
    @assert length(_listOfCars) > 0

    listOfCars = _listOfCars
    drone = _drone

    # Define graph as incidence list representation
    problemGraph =  inclist(Vector{Vertex}(),ExEdge{Vertex},is_directed=true)

    return new(listOfCars,drone,_maxCarSpeed, max(_maxCarSpeed,_drone.speed),length(listOfCars),0,0,Vertex(),Vertex(),Vertex[],Inf,problemGraph,Vector{Tuple{Int,Int}}(),Vector{Int}())
  end
end


function setupGraph(gs::GraphSolution)

  # First add start and goal of drone as vertices
  gs.nVertices += 1
  gs.startVertex = Vertex(gs.nVertices,drone.startPos,0.0,0)
  add_vertex!(gs.problemGraph,gs.startVertex)
  
  gs.nVertices += 1
  gs.goalVertex = Vertex(gs.nVertices,drone.goalPos,Inf,0)
  add_vertex!(gs.problemGraph,gs.goalVertex)
  

  # Now add vertices for each CarRoute
  for car in gs.listOfCars
    thisCarRoute = car.route

    # Add car route nodes AND coasting edges
    gs.nVertices += 1
    prevVert = Vertex(gs.nVertices,thisCarRoute.listOfPoints[1],thisCarRoute.listOfTimes[1],car.idx)
    add_vertex!(gs.problemGraph,prevVert)
    carStartVert = gs.nVertices

    for i = 2 : length(thisCarRoute.listOfPoints)

      # Add vertex corresponding to time-point for car route
      gs.nVertices += 1
      routeVert = Vertex(gs.nVertices,thisCarRoute.listOfPoints[i],thisCarRoute.listOfTimes[i],car.idx)
      add_vertex!(gs.problemGraph,routeVert)

    end
    carEndVert = gs.nVertices
    push!(gs.carIdxRange,(carStartVert,carEndVert))
  end
  
end

function solve(gs::GraphSolution,first_plan::Bool)

  # Setup the goal visitor
  goalVertexIdx = gs.goalVertex.idx
  startVertexIdx = gs.startVertex.idx

  # Edge property inspector - just returns the distance
  basic_weight_inspector = AttributeEdgePropertyInspector{Float64}("weight")


  # Solve Dijkstra Shortest Paths using the inspector and visitor
  #dijkstra_soln = dijkstra_shortest_paths(gs.problemGraph, basic_weight_inspector, [gs.startVertex], visitor=GoalVisitorImplicit(gs))

  heuristic(n) = astar_heuristic(gs,n)

  if first_plan
    path_soln = astar_shortest_path(gs.problemGraph, basic_weight_inspector, gs.startVertex, GoalVisitorImplicit(gs),heuristic)
  else
    path_soln = astar_shortest_path(gs.problemGraph, basic_weight_inspector, gs.startVertex, GoalReplanVisitor(gs),heuristic)
  end


  # Check if the goalvertex has a parent, otherwise report error
  if path_soln.dists[goalVertexIdx] == Inf
    throw(ErrorException("Destination vertex not reachable from source!"))
  end

  # Now use the solution to back out the path from the goal
  solnPathQueue::Deque{Vertex} = Deque{Vertex}()

  unshift!(solnPathQueue,gs.goalVertex)
  currVertexIdx = goalVertexIdx

  while currVertexIdx != startVertexIdx
    prevVertexIdx = path_soln.parent_indices[currVertexIdx]
    unshift!(solnPathQueue,gs.problemGraph.vertices[prevVertexIdx])
    currVertexIdx = prevVertexIdx
  end

  empty!(gs.currSolnPath)
  for v in solnPathQueue
    push!(gs.currSolnPath,v)
  end

  # Get solution cost
  gs.currSolnCost = path_soln.dists[goalVertexIdx]

end

function updateDroneWithCurrSoln(gs::GraphSolution)

  # Take the current solution and put it in the dronePlan
  clear(gs.drone.dronePlan)
  lenSoln::Int = length(gs.currSolnPath)


  for idx = 1 : lenSoln - 1
    # Get pair of iterables
    sp = gs.currSolnPath[idx]
    nsp = gs.currSolnPath[idx+1]

    thisPos = sp.pos
    thisTime = sp.timeVal

    # Now a bunch of if else-s for the vehicle vehicle idx
    if sp.vehicleIdx == nsp.vehicleIdx
      thisIdx = sp.vehicleIdx
    else
      # If both are diff then it is flying between diff vehicles
      thisIdx = 0
    end

    insert(gs.drone.dronePlan,thisPos,thisIdx,thisTime)

  end

  # Insert the last point on the path by updating the time accordingly
  lastDist = pointDist(gs.drone.dronePlan.listOfPoints[end],gs.drone.goalPos)
  lastTime = gs.drone.dronePlan.listOfTimes[end] + lastDist / gs.drone.speed
  insert(gs.drone.dronePlan,gs.drone.goalPos, 0, lastTime)

  gs.drone.hasPlan = true

end


# function replanWithDelays(gs::GraphSolution, delayDict::Dict, currVertex::Vertex)

#   # delayDict : Mapping from car indices to tuples of (route_arc_source,delay)
#   # currVertex : The current vertex along the solution path at which the agent is currently present

  

#   gs.startVertex = currVertex

#   basic_weight_inspector = AttributeEdgePropertyInspector{Float64}("weight")

#   #print(gs.startVertex)

#   goalVertexIdx = gs.goalVertex.idx
#   startVertexIdx = gs.startVertex.idx

#   # Solve Dijkstra Shortest Paths using the inspector and visitor
#   # dijkstra_soln = dijkstra_shortest_paths(gs.problemGraph, basic_weight_inspector, [gs.startVertex], visitor=GoalReplanVisitor(gs))

#   heuristic(n) = astar_heuristic(gs,n)
#   dijkstra_soln = astar_shortest_path(gs.problemGraph, basic_weight_inspector, gs.startVertex, GoalReplanVisitor(gs),heuristic)

#   # Check if the goalvertex has a parent, otherwise report error
#   if dijkstra_soln.dists[goalVertexIdx] == Inf
#     throw(ErrorException("Destination vertex not reachable from source!"))
#   end

#   empty!(gs.currSolnPath)

#   # Now use the solution to back out the path from the goal
#   push!(gs.currSolnPath,gs.goalVertex)
#   currVertexIdx = goalVertexIdx

#   while currVertexIdx != startVertexIdx
#     prevVertexIdx = dijkstra_soln.parent_indices[currVertexIdx]
#     push!(gs.currSolnPath,gs.problemGraph.vertices[prevVertexIdx])
#     currVertexIdx = prevVertexIdx
#   end

#   # Get solution cost
#   gs.currSolnCost = dijkstra_soln.dists[goalVertexIdx]
# end


function updateGraphWithDelay(gs::GraphSolution, delayDict::Dict)

  # delayDict : Mapping from car indices to tuples of (route_arc_source,delay)
  gs.firstRouteIdxAltered = fill(gs.nVertices+1,gs.nCars)

  # Iterate over all given delays and propagate them to all dependent nodes
  # using the car Index range 
  for delayVal in delayDict

    car_idx = delayVal[1]
    relVertIdx,delay = delayVal[2]

    absVertIdx = gs.carIdxRange[car_idx][1] + relVertIdx - 1
    gs.firstRouteIdxAltered[car_idx] = absVertIdx
    
    # now just modify edge weight between absvertidx and (absvertidx + 1)
    if isempty(out_edges(gs.problemGraph.vertices[absVertIdx],gs.problemGraph)) == false
      outedge = out_edges(gs.problemGraph.vertices[absVertIdx],gs.problemGraph)[1]
      outedge.attributes["weight"] = coastEdgeWeight(gs.problemGraph.vertices[absVertIdx],gs.problemGraph.vertices[absVertIdx+1])
    end


    # PROPAGATE the time delay downstream

    absVertIdx += 1
    gs.problemGraph.vertices[absVertIdx].timeVal += delay
    gs.problemGraph.vertices[absVertIdx].settled = false

    # Propagate delay to downstream nodes
    while absVertIdx < gs.carIdxRange[car_idx][2]
      absVertIdx += 1
      gs.problemGraph.vertices[absVertIdx].timeVal += delay
      gs.problemGraph.vertices[absVertIdx].settled = false
    end

  end

end


function astar_heuristic(gs::GraphSolution, v::Vertex)

  dist = pointDist(v.pos,gs.goalVertex.pos)
  val = TIME_COEFFICIENT*dist/gs.maxCarSpeed
  return val
end


mutable struct GoalVisitorSimple <: AbstractDijkstraVisitor
  goalVertex::Vertex
end

function Graphs.include_vertex!(vis::GoalVisitorSimple,u,v,d)
  return v != vis.goalVertex
end


mutable struct GoalVisitorImplicit <: AbstractDijkstraVisitor
  gsObj::GraphSolution
end


function Graphs.include_vertex!(vis::GoalVisitorImplicit,u,v,d)

  # Out_Edges from start have already been processed
  if v == vis.gsObj.goalVertex
    return false
  end

  
  # First add coasting edge to next vertex in car
  veh_idx = v.vehicleIdx
  ver_idx = v.idx

  if veh_idx > 0

    # Copy over the parent's passed Set because that is the current path to
    # vertex v is the best one discovered so far 
    v.passedCars = deepcopy(u.passedCars)

    # If hopping off car, add car idx to passedCars
    # So V does not search here on out
    if u.vehicleIdx > 0 && u.vehicleIdx != v.vehicleIdx
      push!(v.passedCars,u.vehicleIdx)
    end

    veh_ver_range = vis.gsObj.carIdxRange[veh_idx]

    # Add coast edge if it is not the last edge in the route
    if ver_idx < veh_ver_range[2]
      vis.gsObj.nEdges += 1
      routeEdge = ExEdge(vis.gsObj.nEdges,v,vis.gsObj.problemGraph.vertices[ver_idx+1],)
      routeEdge.attributes["weight"] = coastEdgeWeight(v,vis.gsObj.problemGraph.vertices[ver_idx+1])
      add_edge!(vis.gsObj.problemGraph,routeEdge)
    end
  end

  # Add flight edge to goal
  vis.gsObj.nEdges += 1
  flightEdge = ExEdge(vis.gsObj.nEdges, v, vis.gsObj.goalVertex,)
  flightEdge.attributes["weight"] = flightEdgeWeight(v,vis.gsObj.goalVertex,vis.gsObj.drone.speed)
  flightEdge.attributes["distance"] = pointDist(v.pos,vis.gsObj.goalVertex.pos)
  add_edge!(vis.gsObj.problemGraph,flightEdge)

  # Now add flight edges to other cars
  for i = 1 : vis.gsObj.nCars
    if i in v.passedCars || i == veh_idx
      continue
    end

    thisCarRange = vis.gsObj.carIdxRange[i]
    for next_vidx = thisCarRange[2] : -1 : thisCarRange[1]

      pot_next_v = vis.gsObj.problemGraph.vertices[next_vidx]

      # going backwards in time, so if time reduced, break
      if pot_next_v.timeVal < v.timeVal
        break
      end

      if vis.gsObj.drone.speed*(pot_next_v.timeVal - v.timeVal) > pointDist(pot_next_v.pos,v.pos)
        vis.gsObj.nEdges += 1
        flightEdge = ExEdge(vis.gsObj.nEdges, v, pot_next_v,)
        flightEdge.attributes["weight"] = flightEdgeWeight(v,pot_next_v,vis.gsObj.drone.speed)
        flightEdge.attributes["distance"] = pointDist(v.pos,pot_next_v.pos)
        add_edge!(vis.gsObj.problemGraph,flightEdge)
      end
    end

  end

  v.settled = true

  return true
end


mutable struct GoalReplanVisitor <: AbstractDijkstraVisitor
  gsObj::GraphSolution
end

function Graphs.include_vertex!(vis::GoalReplanVisitor,u,v,d)

  # Out_Edges from start have already been processed
  if v == vis.gsObj.goalVertex
    return false
  end

  veh_idx = v.vehicleIdx
  ver_idx = v.idx

  # TODO : If veh_idx is 0 (in the general case) it is some intra-flight vertex
  # Somewhere in the midst of a flight edge (Need a different data structure to rep. this)
  # In that case hallucinate a new vertex where you are and modify

  if veh_idx > 0
    # This is a route vertex. Its next coast edge would have been created already
    # Unless it was never touched the first time

    # Copy over the parent's passed car Set because that is the current path to
    # vertex v is the best one discovered so far 
    v.passedCars = deepcopy(u.passedCars)

    # If hopping off car, add car idx to passedCars
    # So V does not search here on out
    if u.vehicleIdx > 0 && u.vehicleIdx != veh_idx
      push!(v.passedCars,u.vehicleIdx)
    end

    veh_ver_range = vis.gsObj.carIdxRange[veh_idx]

    # First deal with completely untouched vertices
    # We assume that any expanded vertex would at least have an edge to the goal
    if isempty(out_edges(v,vis.gsObj.problemGraph))

      # Add coasting edges if vehicle vertex
      if ver_idx < veh_ver_range[2]
        vis.gsObj.nEdges += 1
        routeEdge = ExEdge(vis.gsObj.nEdges,v,vis.gsObj.problemGraph.vertices[ver_idx+1],)
        routeEdge.attributes["weight"] = coastEdgeWeight(v,vis.gsObj.problemGraph.vertices[ver_idx+1])
        add_edge!(vis.gsObj.problemGraph,routeEdge)
      end
      
      # Add flight edges to goal
      vis.gsObj.nEdges += 1
      flightEdge = ExEdge(vis.gsObj.nEdges, v, vis.gsObj.goalVertex,)
      flightEdge.attributes["weight"] = flightEdgeWeight(v,vis.gsObj.goalVertex, vis.gsObj.drone.speed)
      add_edge!(vis.gsObj.problemGraph,flightEdge)


      # Add flight edges to valid vertices of other cars
      for i = 1 : vis.gsObj.nCars
        if i == veh_idx || i in v.passedCars
          continue
        end

        thisCarRange = vis.gsObj.carIdxRange[i]
        for next_vidx = thisCarRange[2] : -1 : thisCarRange[1]

          pot_next_v = vis.gsObj.problemGraph.vertices[next_vidx]

          # going backwards in time, so if time reduced, break
          if pot_next_v.timeVal < v.timeVal
            break
          end

          if vis.gsObj.drone.speed*(pot_next_v.timeVal - v.timeVal) > pointDist(pot_next_v.pos,v.pos)
            vis.gsObj.nEdges += 1
            flightEdge = ExEdge(vis.gsObj.nEdges, v, pot_next_v,)
            flightEdge.attributes["weight"] = flightEdgeWeight(v,pot_next_v,vis.gsObj.drone.speed)
            flightEdge.attributes["distance"] = pointDist(v.pos,pot_next_v.pos)
            add_edge!(vis.gsObj.problemGraph,flightEdge)
          end
        end
      end
    else

      # Has been expanded before in previous search

      # Note the vertices for each car that have been altered
      modifiedCarIdxs = fill(Set{Int}(),vis.gsObj.nCars)

      # Skip over coast edge
      # TODO : 2:end assumption does NOT work for last route vertex
      # For last route vertex, 1st edge is goal edge which is unchanged so 2:end may be workable here?
      for outedge in out_edges(v,vis.gsObj.problemGraph)[2:end] 

        tgt = target(outedge,vis.gsObj.problemGraph)

        # TODO : Modify if either unsettled
        # If edge no longer exists, set weight to infinity
        # Compute set of all route vertices 
        if !v.settled || !tgt.settled
          if tgt.vehicleIdx > 0
            push!(modifiedCarIdxs[tgt.vehicleIdx],tgt.idx)
          end
          if vis.gsObj.drone.speed*(tgt.timeVal - v.timeVal) > outedge.attributes["distance"]
            outedge.attributes["weight"] = flightEdgeWeight(v,tgt,vis.gsObj.drone.speed)
          else
            outedge.attributes["weight"] = Inf
          end
        end
      end

      # Now check route vertices for each car that are unsettled but which v did not have edges to before
      # Add edges if valid
      for car_idx = 1:vis.gsObj.nCars

        if car_idx == v.idx || car_idx in v.passedCars
          continue
        end

        # Find the set of altered vertices, and only modify edges to those not already
        # handled among out_edges
        alteredRouteIdxs = Set{Int}(vis.gsObj.firstRouteIdxAltered[car_idx] : vis.gsObj.carIdxRange[car_idx][2])
        idxsToCheck = setdiff(modifiedCarIdxs[car_idx],alteredRouteIdxs)

        # Now check for new edges for the rest of the route
        for idxC in idxsToCheck

          tgt = vis.gsObj.problemGraph.vertices[idxC]

          # We know tgt is unsettled so can directly check for edge validity
          if vis.gsObj.drone.speed*(tgt.timeVal - v.timeVal) > pointDist(v.pos,tgt.pos)
            vis.gsObj.nEdges += 1
            flightEdge = ExEdge(vis.gsObj.nEdges, v, tgt,)
            flightEdge.attributes["weight"] = flightEdgeWeight(v,tgt,vis.gsObj.drone.speed)
            flightEdge.attributes["distance"] = pointDist(v.pos,tgt.pos)
            add_edge!(vis.gsObj.problemGraph,flightEdge)
          end #endif
        end #endfor
      end #endfor
    end #endelse

  end

  # Vertex is settled again
  v.settled = true

  return true

end

