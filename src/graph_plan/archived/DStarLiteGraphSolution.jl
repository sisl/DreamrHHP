using Graphs
using DataStructures
using IJulia
include("Types.jl")
include("Solution.jl")
include("GraphHelpers.jl")
include("DStarLite.jl")

# CURRENTLY ASSUME SINGLE DRONE
mutable struct DStarLiteGraphSolution <: Solution

  # Member variables
  listOfCars::Vector{Car}
  drone::Drone
  maxCarSpeed::Float64
  maxSystemSpeed::Float64
  nCars::Int
  nVertices::Int
  nEdges::Int
  startVertex::DStarVertex
  goalVertex::DStarVertex
  currSolnPath::Vector{DStarVertex}
  currSolnCost::Float64
  problemGraph::Graph{DStarVertex,ExEdge{DStarVertex}}
  carIdxRange::Vector{Tuple{Int,Int}}
  firstRouteIdxAltered::Vector{Int}


  function DStarLiteGraphSolution(_listOfCars::Vector{Car},
                                  _drone::Drone,
                                  _maxCarSpeed::Float64)
    @assert length(_listOfCars) > 0

    listOfCars = _listOfCars
    drone = _drone

    # Define graph as incidence list representation
    problemGraph =  graph(Vector{DStarVertex}(),Vector{ExEdge{DStarVertex}}(),is_directed=true)

    return new(listOfCars,drone,_maxCarSpeed, max(_maxCarSpeed,drone.speed),length(listOfCars),0,0,DStarVertex(),DStarVertex(),DStarVertex[],Inf,problemGraph,Vector{Tuple{Int,Int}}(),Vector{Int}())
  end
end


function setupGraph(gs::DStarLiteGraphSolution)

  # First add start and goal of drone as vertices
  gs.nVertices += 1
  gs.startVertex = DStarVertex(gs.nVertices,drone.startPos,0.0,0)
  add_vertex!(gs.problemGraph,gs.startVertex)
  
  gs.nVertices += 1
  gs.goalVertex = DStarVertex(gs.nVertices,drone.goalPos,Inf,0)
  add_vertex!(gs.problemGraph,gs.goalVertex)
  

  # Now add vertices for each CarRoute
  for car in gs.listOfCars
    thisCarRoute = car.route

    # Add car route nodes AND coasting edges
    gs.nVertices += 1
    prevVert::DStarVertex = DStarVertex(gs.nVertices,thisCarRoute.listOfPoints[1],thisCarRoute.listOfTimes[1],car.idx)
    add_vertex!(gs.problemGraph,prevVert)
    carStartVert::Int = gs.nVertices

    for i = 2 : length(thisCarRoute.listOfPoints)

      # Add vertex corresponding to time-point for car route
      gs.nVertices += 1
      routeVert::DStarVertex = DStarVertex(gs.nVertices,thisCarRoute.listOfPoints[i],thisCarRoute.listOfTimes[i],car.idx)
      add_vertex!(gs.problemGraph,routeVert)

    end
    carEndVert::Int = gs.nVertices
    push!(gs.carIdxRange,(carStartVert,carEndVert))
  end
  
end


function solve(gs::DStarLiteGraphSolution,first_plan::Bool,prev_start=nothing,state=nothing)

  basic_weight_inspector = AttributeEdgePropertyInspector{Float64}("weight")
  goalVertexIdx = gs.goalVertex.idx
  startVertexIdx = gs.startVertex.idx

  heuristic(n) = dstar_heuristic(gs,startVertexIdx,n)

  if first_plan
    dstar_soln = dstarlite_solve(gs.problemGraph,basic_weight_inspector,gs.startVertex,gs.goalVertex,
      ImplicitDStarLitePredecessorGenerator(gs),heuristic)
  else
    dstar_soln = dstarlite_resolve(gs.problemGraph,basic_weight_inspector,prev_start,gs.startVertex,gs.goalVertex,
      ReplanDStarLitePredecessorGenerator(),state,heuristic)
  end

  if dstar_soln.rhsvalue[startVertexIdx] == Inf
    throw(ErrorException("Destination vertex not reachable from source!"))
  end

  empty!(gs.currSolnPath)
  push!(gs.currSolnPath,gs.startVertex)
  currVertexIdx = startVertexIdx

  while currVertexIdx != goalVertexIdx
    minsuccval = Inf
    minsuccidx = -1
    for oe in out_edges(gs.problemGraph.vertices[currVertexIdx],gs.problemGraph)
      succidx = target(oe,gs.problemGraph).idx
      if oe.attributes["weight"] + dstar_soln.gvalue[succidx] < minsuccval
        minsuccval = oe.attributes["weight"] + dstar_soln.rhsvalue[succidx]
        minsuccidx = succidx
      end
    end
    push!(gs.currSolnPath,gs.problemGraph.vertices[minsuccidx])
    currVertexIdx = minsuccidx
  end

  gs.currSolnCost = dstar_soln.gvalue[startVertexIdx]

  return dstar_soln

end


function updateDroneWithCurrSoln(gs::DStarLiteGraphSolution)

  # TODO : This is exactly the same as for GraphSolution
  # Take the current solution and put it in the dronePlan
  clear(gs.drone.dronePlan)
  lenSoln = length(gs.currSolnPath)
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


function updateGraphWithDelay(gs::DStarLiteGraphSolution, delayDict::Dict, dstar_state::DStarLiteStates)

  # ASSUME THAT ALL DELAYS ARE IN FUTURE

  # TODO : So far we are setting deleted edge weights to infinity instead of deleting
  # May have to figure out how to handle this in the long run
  # Batched edge removals?

  # delayDict : Mapping from car indices to tuples of (route_arc_source,delay)
  gs.firstRouteIdxAltered = fill(gs.nVertices+1,gs.nCars)

  verticesToUpdate::Vector{Bool} = fill(false,gs.nVertices)

  # Iterate over all given delays and propagate them to all dependent nodes
  # using the car Index range 
  for delayVal in delayDict

    car_idx::Int = delayVal[1]
    relVertIdx,delay = delayVal[2]

    absVertIdx::Int = gs.carIdxRange[car_idx][1] + relVertIdx - 1
    gs.firstRouteIdxAltered[car_idx] = absVertIdx

    firstAlteredVert::DStarVertex = gs.problemGraph.vertices[absVertIdx]


    # now just modify edge weight between absvertidx and (absvertidx + 1)
    if out_degree(firstAlteredVert,gs.problemGraph) > 0
      # CAN NO LONGER ASSUME THAT FIRST outedge IS coast
      # CAN we find index of next vertex in nbrs list and use that index for edge?
      # print(out_neighbors(firstAlteredVert,gs.problemGraph))
      # print(absVertIdx+1)
      route_nbr = gs.problemGraph.vertices[absVertIdx+1]
      nbr_idx::Int = findfirst(out_neighbors(firstAlteredVert,gs.problemGraph), route_nbr)
      outedge = out_edges(gs.problemGraph.vertices[absVertIdx],gs.problemGraph)[nbr_idx]

      if target(outedge,gs.problemGraph).idx != absVertIdx+1
        throw(ErrorException("Out_nbr index does not correspond to outedge index"))
      end

      outedge.attributes["weight"] = coastEdgeWeight(gs.problemGraph.vertices[absVertIdx],gs.problemGraph.vertices[absVertIdx+1])
      verticesToUpdate[absVertIdx] = true
    end


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

  # UPDATE FLIGHT EDGES!
  # Now iterate over all vertices
  # If untouched, then look for new edges and reinsertion in Queue
  # If touched and unsettled, then reconsider all edges
  # If touched and settled, then only consider edges to unsettled 

  for v in gs.problemGraph.vertices

    if v.timeVal < gs.startVertex.timeVal
      continue
    end

    toUpdate::Bool = false

    if out_degree(v,gs.problemGraph) == 0
      # Previously untouched vertex
      # Loop through all car vertices and add new ones if valid
      for car_idx = 1:gs.nCars

        if car_idx == v.idx
          continue
        end

        # Go through all modified car vertices
        # And check if edge now exists where it did not before
        # NEED DSTAR_SOLN OBJECT!
        for mod_idx = gs.carIdxRange[car_idx][2] : -1 : gs.firstRouteIdxAltered[car_idx]

          # If target unsettled then there can be a potentially better edge between
          # this index and the unsettled target
          # The effective c_old here is infinity so. can update accordingly
          mod_tgt = gs.problemGraph.vertices[mod_idx]

          if mod_tgt.timeVal < v.timeVal
            break
          end

          if mod_tgt.settled == false && gs.drone.speed*(mod_tgt.timeVal - v.timeVal) > pointDist(v.pos,mod_tgt.pos)
            gs.nEdges += 1
            flightEdge::ExEdge{DStarVertex} = ExEdge(gs.nEdges, v, mod_tgt,)
            flightEdge.attributes["weight"] = flightEdgeWeight(v, mod_tgt, gs.drone.speed)
            flightEdge.attributes["distance"] = pointDist(v.pos, mod_tgt.pos)
            add_edge!(gs.problemGraph,flightEdge)

            # Do the updation step knowing that c_old > c(u,v)
            if v != gs.goalVertex 
              dstar_state.rhsvalue[v.idx] = min(dstar_state.rhsvalue[v.idx], flightEdge.attributes["weight"] + dstar_state.gvalue[mod_tgt.idx])
            end
            toUpdate = true
          end
        end
      end

    else # if out_degree
      # Existing edges - update or delete them
      modifiedCarIdxs = fill(Set{Int}(),gs.nCars)


      for outedge in out_edges(v,gs.problemGraph)
        mod_tgt = target(outedge,gs.problemGraph)

        # If not coast edge and either of them unsettled
        if mod_tgt.vehicleIdx != v.vehicleIdx && (!v.settled || !mod_tgt.settled)
          if mod_tgt.vehicleIdx > 0
            push!(modifiedCarIdxs[mod_tgt.vehicleIdx],mod_tgt.idx)
          end
          toUpdate = true
          oldweight = outedge.attributes["weight"]
          if gs.drone.speed*(mod_tgt.timeVal - v.timeVal) > outedge.attributes["distance"]
            # Recompute weights and compare to old
            outedge.attributes["weight"] = flightEdgeWeight(v,mod_tgt,gs.drone.speed)
            if oldweight > outedge.attributes["weight"]
              if v != gs.goalVertex
                dstar_state.rhsvalue[v.idx] = min(dstar_state.rhsvalue[v.idx], outedge.attributes["weight"]+dstar_state.gvalue[mod_tgt.idx])
              end
            elseif dstar_state.rhsvalue[v.idx] == oldweight + dstar_state.gvalue[mod_tgt.idx]
              if v!= gs.goalVertex
                minrhsval = Inf
                for succ_edge in out_edges(v,gs.problemGraph)
                  minrhsval = min(minrhsval, succ_edge.attributes["weight"] + dstar_state.gvalue[target(succ_edge,gs.problemGraph).idx])
                end
                dstar_state.rhsvalue[v.idx] = minrhsval
              end
            end
          else
            # An existing edge gets deleted. Assume c(u,v) = inf > c_old
            outedge.attributes["weight"] = Inf
            if dstar_state.rhsvalue[v.idx] == oldweight + dstar_state.gvalue[v.idx]
              if v!= gs.goalVertex
                minrhsval = Inf
                for succ_edge in out_edges(v,gs.problemGraph)
                  minrhsval = min(minrhsval, succ_edge.attributes["weight"] + dstar_state.gvalue[target(succ_edge,gs.problemGraph).idx])
                end
                dstar_state.rhsvalue[v.idx] = minrhsval
              end
            end
          end
        end
      end

      # Now check other unsettled vertices for potential new connections
      for car_idx = 1:gs.nCars

        if car_idx == v.idx
          continue
        end

        # Find the set of altered vertices, and only modify edges to those not already
        # handled among out_edges
        alteredRouteIdxs = Set{Int}(gs.firstRouteIdxAltered[car_idx] : gs.carIdxRange[car_idx][2])
        idxsToCheck = setdiff(modifiedCarIdxs[car_idx],alteredRouteIdxs)

        # Now check for new edges for the rest of the route
        for idxC in idxsToCheck

          mod_tgt = gs.problemGraph.vertices[idxC]

          if gs.drone.speed*(mod_tgt.timeVal - v.timeVal) > pointDist(v.pos,mod_tgt.pos)
            gs.nEdges += 1
            flightEdge = ExEdge(gs.nEdges, v, mod_tgt,)
            flightEdge.attributes["weight"] = flightEdgeWeight(v,mod_tgt,gs.drone.speed)
            flightEdge.attributes["distance"] = pointDist(v.pos,mod_tgt.pos)
            add_edge!(gs.problemGraph,flightEdge)

            # now edge effectively exists between v and mod_tgt
            if v != gs.goalVertex
              dstar_state.rhsvalue[v.idx] = min(dstar_state.rhsvalue[v.idx], flightEdge.attributes["weight"] + dstar_state.gvalue[mod_tgt.idx])
            end
          end #endif
        end
      end
    end

    if toUpdate
      verticesToUpdate[v.idx] = true
    end

  end

  heuristic(n) = dstar_heuristic(gs,gs.startVertex.idx,n)

  for i = 1:gs.nVertices
    if verticesToUpdate[i]
      update_vertex!(dstar_state, i, heuristic)
    end
  end
end


function dstar_heuristic(gs::DStarLiteGraphSolution, idx1::Int, idx2::Int)

  v1 = gs.problemGraph.vertices[idx1]
  v2 = gs.problemGraph.vertices[idx2]
  dist = pointDist(v1.pos,v2.pos)
  val = TIME_COEFFICIENT*dist/gs.maxCarSpeed
  return val
end


mutable struct ImplicitDStarLitePredecessorGenerator <: AbstractDStarLitePredecessorGenerator
  gsObj::DStarLiteGraphSolution
end

function generate_predecessors!(gen::ImplicitDStarLitePredecessorGenerator, v::DStarVertex)

  # Return immediately if start vertex
  if v == gen.gsObj.startVertex
    return
  end

  # TODO - COAST EDGE MAY NO LONGER BE FIRST OUTEDGE!!!

  veh_idx::Int = v.vehicleIdx
  ver_idx::Int = v.idx

  if veh_idx > 0

    # Is a car route vertex - add preceding coast edge
    veh_ver_range = gen.gsObj.carIdxRange[veh_idx]

    if ver_idx > veh_ver_range[1]
      gen.gsObj.nEdges += 1
      routeEdge = ExEdge(gen.gsObj.nEdges,gen.gsObj.problemGraph.vertices[ver_idx - 1],v,)
      routeEdge.attributes["weight"] = coastEdgeWeight(gen.gsObj.problemGraph.vertices[ver_idx - 1],v)
      add_edge!(gen.gsObj.problemGraph,routeEdge)
    end
  end

  # First add flight edge from start if reachable
  if gen.gsObj.drone.speed*(v.timeVal) > pointDist(gen.gsObj.startVertex.pos, v.pos)
    gen.gsObj.nEdges += 1
    startGoalEdge::ExEdge{DStarVertex} = ExEdge(gen.gsObj.nEdges, gen.gsObj.startVertex, v, )
    startGoalEdge.attributes["weight"] = flightEdgeWeight(gen.gsObj.startVertex, v, gen.gsObj.drone.speed)
    startGoalEdge.attributes["distance"] = pointDist(gen.gsObj.startVertex.pos, v.pos)
    add_edge!(gen.gsObj.problemGraph,startGoalEdge)
  end

  # Now cycle through car vertices and add from them to goal
  # if they are reachable from start 
  # ASSUMPTION - 1 and 2 are start/goal
  for i = 1: gen.gsObj.nCars

    thisCarRange = gen.gsObj.carIdxRange[i]

    for prev_vidx = thisCarRange[1] : thisCarRange[2]

      pot_prev_v = gen.gsObj.problemGraph.vertices[prev_vidx]

      if pot_prev_v.timeVal > v.timeVal
        break
      end

      if pot_prev_v.reachable == no
        continue
      end

      # Check if potential predecessor is even reachable by start
      if pot_prev_v.reachable == yes || gen.gsObj.maxSystemSpeed*(pot_prev_v.timeVal - gen.gsObj.startVertex.timeVal) > pointDist(pot_prev_v.pos,gen.gsObj.startVertex.pos)
        pot_prev_v.reachable = yes

        if gen.gsObj.drone.speed*(v.timeVal - pot_prev_v.timeVal) > pointDist(v.pos, pot_prev_v.pos)
          gen.gsObj.nEdges += 1
          flightEdge::ExEdge{DStarVertex} = ExEdge(gen.gsObj.nEdges, pot_prev_v, v,)
          flightEdge.attributes["weight"] = flightEdgeWeight(pot_prev_v, v, gen.gsObj.drone.speed)
          flightEdge.attributes["distance"] = pointDist(pot_prev_v.pos, v.pos)
          add_edge!(gen.gsObj.problemGraph,flightEdge)
        end
      end
    end
  end
end


mutable struct ReplanDStarLitePredecessorGenerator <: AbstractDStarLitePredecessorGenerator
end


# Implicitly update edge costs 
generate_predecessors!(gen::ReplanDStarLitePredecessorGenerator, v::DStarVertex) = nothing







