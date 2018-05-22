include("Types.jl")
include("Solution.jl")

# Definitions required for graph operations
mutable struct Vertex
  idx::Int
  pos::Point
  timeVal::Float64 # Use time 0.0 for start and inf for goal
  vehicleIdx::Int
  passedCars::Set{Int}
  settled::Bool
  function Vertex(idx::Int,_pos::Point,_timeVal::Float64,_vehicleIdx::Int)
    pos = deepcopy(_pos)
    return new(idx,pos,_timeVal,_vehicleIdx,Set{Int}(),true)
  end
end
Vertex() = Vertex(0,Point(0.0,0.0),0.0,0)

function Graphs.vertex_index(v::Vertex,g::IncidenceList{Vertex,ExEdge{Vertex}})
  return v.idx
end

@enum Reachable yes=1 no=2 unknown=3

mutable struct DStarVertex
  idx::Int
  pos::Point
  timeVal::Float64 # Use time 0.0 for start and inf for goal
  vehicleIdx::Int
  reachable::Reachable
  settled::Bool
  function DStarVertex(_idx::Int,_pos::Point,_timeVal::Float64,_vehicleIdx::Int)
    pos = deepcopy(_pos)
    return new(_idx,pos,_timeVal,_vehicleIdx,unknown,true)
  end
end

DStarVertex() = DStarVertex(0,Point(0.0,0.0),0.0,0)

function Graphs.vertex_index(v::DStarVertex,g::Graph{DStarVertex,ExEdge{DStarVertex}})
  return v.idx
end


function flightEdgeWeight(u::Vertex, v::Vertex, speed::Float64)

  weight = oneStepFlightCost(u.pos,v.pos,u.timeVal,v.timeVal,speed)
  return weight

end

function coastEdgeWeight(u::Vertex, v::Vertex)

  weight = oneStepCoastCost(u.timeVal,v.timeVal)
  return weight

end

function flightEdgeWeight(u::DStarVertex, v::DStarVertex, speed::Float64)

  weight = oneStepFlightCost(u.pos,v.pos,u.timeVal,v.timeVal,speed)
  return weight

end

function coastEdgeWeight(u::DStarVertex, v::DStarVertex)

  weight = oneStepCoastCost(u.timeVal,v.timeVal)
  return weight

end

