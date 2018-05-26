import JSON
using Colors
using Reel

include("../src/Types.jl")
include("../src/GraphSolution.jl")
include("../src/DStarLiteGraphSolution.jl")
include("../src/Utils.jl")


DATAFILE = "../data/basic_set2.json"

dicttxt = ""
  
open(DATAFILE,"r") do f_in
  global dicttxt
  dicttxt = readstring(f_in)
end

probdict = JSON.parse(dicttxt)

drone, listOfCars, maxCarSpeed = constructBasicProblem(probdict)



@time graphSolnObj = GraphSolution(listOfCars, drone, maxCarSpeed)

@time setupGraph(graphSolnObj)

@time solve(graphSolnObj,true)

updateDroneWithCurrSoln(graphSolnObj)
printPlan(graphSolnObj.drone.dronePlan)

PATHVERTEX = 2
timeThresh = graphSolnObj.currSolnPath[PATHVERTEX].timeVal
delayDict = generateRandomDelayAfterTime(graphSolnObj,timeThresh);


updateGraphWithDelay(graphSolnObj, delayDict)
graphSolnObj.startVertex = graphSolnObj.currSolnPath[PATHVERTEX]
@time solve(graphSolnObj,false)
updateDroneWithCurrSoln(graphSolnObj)

printPlan(graphSolnObj.drone.dronePlan)


@time dstarSolnObj = DStarLiteGraphSolution(listOfCars, drone, maxCarSpeed)
@time setupGraph(dstarSolnObj)
@time dstar_state = solve(dstarSolnObj,true)

updateDroneWithCurrSoln(dstarSolnObj)
printPlan(dstarSolnObj.drone.dronePlan)


prev_start = dstarSolnObj.startVertex
dstarSolnObj.startVertex = dstarSolnObj.currSolnPath[PATHVERTEX]
updateGraphWithDelay(dstarSolnObj,delayDict,dstar_state)

@time dstar_state = solve(dstarSolnObj,false,prev_start,dstar_state)

updateDroneWithCurrSoln(dstarSolnObj)
printPlan(dstarSolnObj.drone.dronePlan)



# carColVect = Color[]

# for i = 1 : graphSolnObj.nCars
#   push!(carColVect,RGB(rand(),rand(),rand()))
# end

# comp_prob = constructProblemComposition(graphSolnObj,15cm,carColVect)
# comp_soln = constructSolnComposition(graphSolnObj,15cm,carColVect)
# comp_full = compose(context(),comp_prob,comp_soln)
# comp_full |> PNG("solution.png")

# comp_points = generatePositionsAtTime(graphSolnObj,15cm,carColVect,15.0)

# comp_current = compose(context(),comp_prob,comp_points)

# comp_time(t) = compose(context(),comp_prob,generatePositionsAtTime(graphSolnObj,15cm,carColVect,Float64(t)))
# routeEndTime = Int(ceil(graphSolnObj.drone.dronePlan.listOfTimes[end]))
# film = roll(map(comp_time, 1:routeEndTime), fps=0.5)

# println(routeEndTime)

# comp_points = generatePositionsAtTime(graphSolnObj,15cm,carColVect,routeEndTime/2)
# comp_current = compose(context(),comp_prob,comp_points)

# frames = Frames(MIME("image/png"), fps=1)

# for timeVal in graphSolnObj.drone.dronePlan.listOfTimes
#   comp_points = generatePositionsAtTime(graphSolnObj,15cm,carColVect,timeVal)
#   comp_current = compose(context(),comp_points,comp_prob)
#   comp_current |> PNG(string("test",timeVal,".png"))
#   push!(frames,comp_current)
# end

# write("output.mp4",frames)

# # Now choose the timepoint at which delay happens
