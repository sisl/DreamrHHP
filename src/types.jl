using DataStructures

# Data structure for coordinates
mutable struct Point
  x::Float64
  y::Float64
end
Point() = Point(0.0,0.0)

function pointDist(p1::Point, p2::Point)
  distSq = (p1.x - p2.x)^2 + (p1.y - p2.y)^2
  return sqrt(distSq)
end

function point_norm(p::Point)
  return sqrt(p.x^2 + p.y^2)
end

function interpolate(p1::Point,p2::Point,frac::Float64)

  xval = p1.x + frac*(p2.x-p1.x)
  yval = p1.y + frac*(p2.y-p1.y)

  return Point(xval,yval)

end

# Point with an associated time
struct TimePoint
  point::Point
  timeVal::Float64
end


mutable struct CarRoute
  listOfPoints::Array{Point,1} # Set of waypoints for Car
  listOfTimes::Array{Float64,1} # Time corresponding to car point

  function CarRoute(_listOfPoints::Array{Point,1},
                    _listOfTimes::Array{Float64,1})
    @assert length(_listOfPoints) == length(_listOfTimes)
    listOfPoints = deepcopy(_listOfPoints)
    listOfTimes = deepcopy(_listOfTimes)
    return new(listOfPoints,listOfTimes)
  end
end


mutable struct Car
  idx::Int
  capacity::Int
  route::CarRoute
  cargoDroneIdx::Int # 0 if no drone

  function Car(_idx::Int, _capacity::Int,_route::CarRoute)
    route = deepcopy(_route)
    return new(_idx,_capacity,route,0)
  end
end


mutable struct DronePlan
  listOfPoints::Vector{Point}
  listOfIdxs::Vector{Int} # Vehicle indices for the vehicle for that interval; 0 if self
  listOfTimes::Vector{Float64}

  function DronePlan()
    return new(Vector{Point}(),Vector{Int}(),Vector{Float64}())
  end
end

function clear(d::DronePlan)
  empty!(d.listOfPoints)
  empty!(d.listOfIdxs)
  empty!(d.listOfTimes)
end

function insert(d::DronePlan,pt::Point,idx::Int,timeVal::Float64)
  push!(d.listOfPoints,pt)
  push!(d.listOfIdxs,idx)
  push!(d.listOfTimes,timeVal)
end

function printPlan(d::DronePlan)
  nPoints = length(d.listOfPoints)

  for i in 1 : nPoints
    pt = d.listOfPoints[i]
    println("(",pt.x,",",pt.y,") ; ",d.listOfTimes[i]," ; ",d.listOfIdxs[i])
  end
end



mutable struct Drone
  idx::Int
  speed::Float64
  startPos::Point
  goalPos::Point
  dronePlan::DronePlan
  hasPlan::Bool

  function Drone(_idx::Int, _speed::Float64,
                 _startPos::Point, _goalPos::Point)
    startPos = deepcopy(_startPos)
    goalPos = deepcopy(_goalPos)
    return new(_idx,_speed,startPos,goalPos,DronePlan(),false)
  end
end