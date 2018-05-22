using DataStructures

# Data structure for coordinates
mutable struct Point
    x::Float64
    y::Float64
end
Point() = Point(0.0,0.0)
Point(t::Tuple{Float64,Float64}) = Point(t[1],t[2])

function point_dist(p1::Point, p2::Point)
    distSq = (p1.x - p2.x)^2 + (p1.y - p2.y)^2
    return sqrt(distSq)
end

function point_norm(p::Point)
    return sqrt(p.x^2 + p.2)
end

function interpolate(p1::Point,p2::Point,frac::Float64)

    xval = p1.x + frac*(p2.x-p1.x)
    yval = p1.y + frac*(p2.y-p1.y)

    return Point(xval,yval)

end

TimeStampedPoint = Tuple{Point, Float64}

mutable struct Car
    car_idx::String
    capacity::Int
    route::Dict{String, Tuple{Point,Float64}}
    next_route_idx::Int # From graph car vertices
    cargoDroneIdx::Int # 0 if no drone
    active::Bool
end

function Car(_car_idx::String)
    return Car(_car_idx, 1, Dict{String, TimePoint}(), 0, 0, true)
end


# Any physical characteristics of the drone that are not updated during the problem
struct Drone
    idx::Int
    max_speed::Float64
    max_distance::Float64
    max_energy::Float64
end

Drone() = Drone(0,MAX_SPEED,Inf,Inf)

function Drone(_idx::Int)
    return Drone(_idx, MAX_SPEED, Inf, Inf)
end

function Drone(_idx::Int,_max_speed::Float64)
    return Drone(_idx, _max_speed, Inf, Inf)
end