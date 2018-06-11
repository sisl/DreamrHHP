using DataStructures

# Data structure for coordinates
struct Point
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
    return sqrt(p.x^2 + p.y^2)
end

function equal(p1::Point, p2::Point)
    return (isapprox(p1.x,p2.x) && isapprox(p1.y,p2.y))
end

function interpolate(p1::Point,p2::Point,frac::Float64)

    xval = p1.x + frac*(p2.x-p1.x)
    yval = p1.y + frac*(p2.y-p1.y)

    return Point(xval,yval)

end

mutable struct Car
    route_idx_range::Vector{Int} # Size 2 but needs to be changeable
    cargoDroneIdx::Int # 0 if no drone
    capacity::Int
    active::Bool
end

# Default constructor inactive car - unlikely to be used
function InactiveCar()
    return Car([0,0], 0, 1, false)
end


function Car(idx_range::Vector{Int})
    return Car(idx_range, 0, 1, true)
end


# Any physical characteristics of the drone that are not updated during the problem
struct Drone
    idx::Int
    max_speed::Float64
    max_distance::Float64
    max_energy::Float64
end

Drone() = Drone(0,MAX_DRONE_SPEED,Inf,Inf)

function Drone(_idx::Int)
    return Drone(_idx, MAX_DRONE_SPEED, Inf, Inf)
end

function Drone(_idx::Int,_max_speed::Float64)
    return Drone(_idx, _max_speed, Inf, Inf)
end