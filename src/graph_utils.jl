using Compose
using Colors
using Measures
using Distributions

import JSON
include("Types.jl")
include("GraphSolution.jl")

CAR_RADIUS = 0.0075
DRONE_RADIUS = 0.01

function getListOfPointsPerInterval(p1::Point,t1::Float64,p2::Point,t2::Float64)

  # Assume that t1 < t2
  # Get a list of points from p1 to p2 in intervals of 1 second
  distance = pointDist(p1,p2)
  timeDiff = t2 - t1
  xdir = (p2.x - p1.x)
  ydir = (p2.y - p1.y)

  listOfPoints = Point[]
  listOfTimes = Float64[]

  # Get interpolated set of times and then get corresponding positions
  # Check if t1 is integral, else round up
  if ceil(t1) > t1
    push!(listOfTimes,t1)
  end

  # Append the 1-second interval of times
  append!(listOfTimes,collect(ceil(t1) : 1.0 : floor(t2)))

  # now append the last time if it is not integral
  if floor(t2) < t2
    push!(listOfTimes,t2)
  end

  for timeVal in listOfTimes

    time_factor = (timeVal - t1)/(t2 - t1)
    new_x = p1.x + xdir*time_factor
    new_y = p1.y + ydir*time_factor
    new_pt = Point(new_x,new_y)
    push!(listOfPoints,new_pt)
  end

  return listOfPoints,listOfTimes

end


# interval is the max diff in time between two GPS points for car
# So it is a upper bound on how coarse the problem can be
function constructBasicProblem(probdict::Dict)

  carDict = probdict["cars"]
  droneDict = probdict["drone"]

  # First settle cars
  listOfCars = Car[]

  maxCarSpeed = -Inf

  for i = 1 : carDict["ncars"]

    listOfFullPoints = Point[]
    listOfFullTimes = Float64[]

    listOfPoints = carDict[string(i)]["points"]
    listOfTimes = carDict[string(i)]["times"]

    nPoints = length(listOfPoints)

    for j = 1:nPoints-1
      ptj = Point(listOfPoints[j][1],listOfPoints[j][2])
      ptj1 = Point(listOfPoints[j+1][1],listOfPoints[j+1][2])

      spd = pointDist(ptj,ptj1) / (listOfTimes[j+1] - listOfTimes[j])
      if spd > maxCarSpeed
        maxCarSpeed = spd
      end

      expListPts, expListTimes = getListOfPointsPerInterval(
                                  ptj,listOfTimes[j],ptj1,listOfTimes[j+1])
      append!(listOfFullTimes,expListTimes[1:end-1])
      append!(listOfFullPoints,expListPts[1:end-1])
    end

    # Now append the last time
    push!(listOfFullTimes,listOfTimes[end])
    push!(listOfFullPoints,Point(listOfPoints[end][1], listOfPoints[end][2]))

    thisCarRoute = CarRoute(listOfFullPoints,listOfFullTimes)

    thisCar = Car(i,1,thisCarRoute)

    push!(listOfCars,thisCar)

  end

  # Now work on drone
  startPos = Point(droneDict["startPos"][1],droneDict["startPos"][2])
  goalPos = Point(droneDict["goalPos"][1],droneDict["goalPos"][2])

  thisDrone = Drone(1,droneDict["speed"],startPos,goalPos)

  return thisDrone,listOfCars,maxCarSpeed

end


function plotProbGraphSolution(gs::GraphSolution, width::Any, outfilepref::String)

  set_default_graphic_size(width,width)

  listOfCars = gs.listOfCars
  drone = gs.drone

  car_circles = []

  # First plot rectangles for car routes
  for car in listOfCars

    carColor = RGB(rand(),rand(),rand())

    prev_pt = car.route.listOfPoints[1]
    prev_pt_tuple = (prev_pt.x,prev_pt.y)
    push!(car_circles, compose(context(),circle(prev_pt.x,prev_pt.y,CAR_RADIUS),fill(carColor)))

    for pt in car.route.listOfPoints[2:end]

      pt_tuple = (pt.x,pt.y)
      thisCarCirc = compose(context(),circle(pt.x,pt.y,CAR_RADIUS),fill(carColor))
      push!(car_circles,thisCarCirc)
      thisCarLine = compose(context(),line([prev_pt_tuple,pt_tuple]),stroke(carColor), linewidth(0.5mm))
      push!(car_circles,thisCarLine)

      # copy over
      prev_pt = pt
      prev_pt_tuple = deepcopy(pt_tuple)
    end

    # Use square for last point
    push!(car_circles, compose(context(),rectangle(car.route.listOfPoints[end].x-CAR_RADIUS,car.route.listOfPoints[end].y-CAR_RADIUS,3*CAR_RADIUS,3*CAR_RADIUS),fill(carColor)))
  end

  drone_comp = []

  # First add circles for start and end-points of drones
  push!(drone_comp,compose(context(), circle(drone.startPos.x,drone.startPos.y,DRONE_RADIUS), fill(RGB(0.,0.,0.)) ))
  push!(drone_comp,compose(context(), circle(drone.goalPos.x,drone.goalPos.y,DRONE_RADIUS), fill(RGB(0.,0.,0.)) ))

  composition_prob = compose(context(), car_circles,drone_comp)

  edge_lines_comp = []

  # Now do for the problemGraph
  # Iterate over edges and draw lines from vertex to vertex
  for vert in gs.problemGraph.vertices
    for e in out_edges(vert,gs.problemGraph)
      pos1 = (e.source.pos.x,e.source.pos.y)
      pos2 = (e.target.pos.x,e.target.pos.y)

      thisEdgeLine = compose(context(),line([pos1,pos2]),stroke("black"), linewidth(0.25mm))
      push!(edge_lines_comp,thisEdgeLine)
    end
  end

  composition_graph = compose(context(), car_circles, drone_comp, edge_lines_comp)


  prev_pt_tuple = (drone.dronePlan.listOfPoints[1].x, drone.dronePlan.listOfPoints[1].y)

  for pt in drone.dronePlan.listOfPoints[2:end]

    pt_tuple = (pt.x,pt.y)

    thisDroneLine = compose(context(),line([prev_pt_tuple,pt_tuple]),stroke("black"), linewidth(0.75mm))
    push!(drone_comp,thisDroneLine)

    prev_pt_tuple = deepcopy(pt_tuple)
  end
  

  composition_soln = compose(context(), car_circles,drone_comp)

  # composition_prob |> SVG(string(outfilepref,"prob.svg"))
  # composition_graph |> SVG(string(outfilepref,"graph.svg"))
  # composition_soln |> SVG(string(outfilepref,"soln.svg"))

  return composition_soln
end


function constructProblemComposition(gs::GraphSolution,width::Any,carColors::Vector{Color})

  set_default_graphic_size(width,width)

  listOfCars = gs.listOfCars
  drone = gs.drone

  car_circles = []

  for (i,car) in enumerate(listOfCars)

    carColor = carColors[i]

    prev_pt = car.route.listOfPoints[1]
    prev_pt_tuple = (prev_pt.x,prev_pt.y)
    #push!(car_circles, compose(context(),circle(prev_pt.x,prev_pt.y,CAR_RADIUS),fill(carColor)))

    for pt in car.route.listOfPoints[2:end]

      pt_tuple = (pt.x,pt.y)
      thisCarLine = compose(context(),line([prev_pt_tuple,pt_tuple]),stroke(carColor), linewidth(0.5mm))
      push!(car_circles,thisCarLine)

      # copy over
      prev_pt = pt
      prev_pt_tuple = deepcopy(pt_tuple)
    end

    # Use square for last point
    push!(car_circles, compose(context(),rectangle(car.route.listOfPoints[end].x-CAR_RADIUS,car.route.listOfPoints[end].y-CAR_RADIUS,3*CAR_RADIUS,3*CAR_RADIUS),fill(carColor)))
  end

  drone_comp = []

  # First add circles for start and end-points of drones
  push!(drone_comp,compose(context(), circle(drone.startPos.x,drone.startPos.y,DRONE_RADIUS), fill(RGBA(1.,1.,1.,1.0)) ))
  push!(drone_comp,compose(context(), circle(drone.goalPos.x,drone.goalPos.y,DRONE_RADIUS), fill(RGBA(1.,1.,1.,1.0))))

  composition_prob = compose(context(), car_circles,drone_comp)

end


function constructSolnComposition(gs::GraphSolution,width::Any,carColors::Vector{Color})

  set_default_graphic_size(width,width)

  drone_comp = []

  prev_pt_tuple = (drone.dronePlan.listOfPoints[1].x, drone.dronePlan.listOfPoints[1].y)

  for pt in drone.dronePlan.listOfPoints[2:end]

    pt_tuple = (pt.x,pt.y)

    thisDroneLine = compose(context(),line([prev_pt_tuple,pt_tuple]),stroke("white "), linewidth(0.75mm))
    push!(drone_comp,thisDroneLine)

    prev_pt_tuple = deepcopy(pt_tuple)
  end

  composition_soln = compose(context(),drone_comp)

  return composition_soln

end



function findEnclosingPairSource(timeVal::Float64,timeList::Vector{Float64})

  # Guaranteed it is between first and last

  for i = 1 : length(timeList)-1
    if timeList[i] <= timeVal && timeList[i+1] >= timeVal
      return i
    end
  end

  return length(timeList) - 1
end


function generatePositionsAtTime(gs::GraphSolution,width::Any,carColors::Vector{Color},timePoint::Float64)

  set_default_graphic_size(width,width)

  listOfCars = gs.listOfCars
  drone = gs.drone

  car_drone_circles = []

  for (i,car) in enumerate(listOfCars)

    carColor = carColors[i]

    # Only add circle if it is between first and last time
    if timePoint >= car.route.listOfTimes[1] && timePoint <= car.route.listOfTimes[end]

      idx = findEnclosingPairSource(timePoint,car.route.listOfTimes)

      if abs(timePoint - car.route.listOfTimes[idx]) > eps()
        # It is in-between if idx and idx+1 are different
        if car.route.listOfTimes[idx+1] >  car.route.listOfTimes[idx]
          frac = (timePoint - car.route.listOfTimes[idx]) / (car.route.listOfTimes[idx+1] - car.route.listOfTimes[idx])
          carpt = interpolate(car.route.listOfPoints[idx],car.route.listOfPoints[idx+1],frac)
        else
          carpt = car.route.listOfPoints[idx]
        end
      else
        carpt = car.route.listOfPoints[idx]
      end

      push!(car_drone_circles, compose(context(),circle(carpt.x,carpt.y,CAR_RADIUS),fill(carColor)))
    end

  end

  # Now get drone position
  if timePoint >= drone.dronePlan.listOfTimes[1] && timePoint <= drone.dronePlan.listOfTimes[end]

    idx = findEnclosingPairSource(timePoint,drone.dronePlan.listOfTimes)

    if abs(timePoint - drone.dronePlan.listOfTimes[idx]) > eps()

      if drone.dronePlan.listOfTimes[idx+1] >  drone.dronePlan.listOfTimes[idx]

        frac = (timePoint - drone.dronePlan.listOfTimes[idx]) / (drone.dronePlan.listOfTimes[idx+1] - drone.dronePlan.listOfTimes[idx])
        dronept = interpolate(drone.dronePlan.listOfPoints[idx],drone.dronePlan.listOfPoints[idx+1],frac)
      else
        dronept = drone.dronePlan.listOfPoints[idx]
      end
    else
      dronept = drone.dronePlan.listOfPoints[idx]
    end

    push!(car_drone_circles,compose(context(), circle(dronept.x,dronept.y,DRONE_RADIUS), fill(RGB(1.,1.,1.)) ))
  end

  comp_points = compose(context(),car_drone_circles)

  return comp_points

end




function generateRandomDelayAfterTime(gs::GraphSolution,timeThresh::Float64)

  # Iterate through cars, and for each car choose a route arc
  # With time greater than the threshold time
  delayDict = Dict{Int,Tuple}()

  DELAYRANGE = (5.0,15.0)

  for car_idx = 1 : gs.nCars

    # For each car, find first time after threshold and choose
    # a random route index between that and last-but-1
    route_idx = 0
    for route_idx = gs.carIdxRange[car_idx][1] : gs.carIdxRange[car_idx][2]-1
      if gs.problemGraph.vertices[route_idx].timeVal > timeThresh
        break
      end
    end

    # Add to delayDict if at least 1 route point is ahead of time threshold
    if route_idx < gs.carIdxRange[car_idx][2]
      delayDict[car_idx] = (rand(1 : gs.carIdxRange[car_idx][2]-route_idx), rand(Uniform(DELAYRANGE[1],DELAYRANGE[2])))
    end

  end


  return delayDict
end


