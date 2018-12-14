# Data structure for coordinates
"""
Compact structure for 2D point.
"""
struct Point
    x::Float64
    y::Float64
end
Point() = Point(0.0,0.0)
Point(v::V) where {V <: AbstractVector} = Point(v[1], v[2])
Point(t::Tuple{Float64,Float64}) = Point(t[1], t[2])


function point_dist(p1::Point, p2::Point, l::Real=2)
    s = SVector{2,Float64}(p1.x-p2.x, p1.y-p2.y)
    return norm(s,l)
end

function point_norm(p::Point, l::Real=2)
    s = SVector{2,Float64}(p.x,p.y)
    return norm(s,l)
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
    route_idx_range::MVector{2, Int} # Size 2 but needs to be changeable
    cargo_drone_idx::Int # 0 if no drone
    capacity::Int
    active::Bool
end

# Default constructor for inactive car - unlikely to be used
InactiveCar() = Car([0,0], 0, 1,false)

# Constuctor for newly introduced car
function Car(idx_range::V) where {V <: AbstractVector{Int}} 
    @assert length(idx_range) == 2 "Index Range should be of length 2"
    return Car(idx_range, 0, 1, true)
end


# Any physical characteristics of the drone that are not updated during the problem
struct Drone
    idx::Int
    max_speed::Float64
    max_distance::Float64
    max_energy::Float64
end

Drone(idx::Int,max_speed::Float64) = Drone(idx, max_speed, Inf, Inf)