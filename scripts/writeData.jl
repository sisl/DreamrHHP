import JSON


carsDict = Dict("1" => Dict("points" => [(0.03,0.08), (0.4,0.18), (0.3,0.45)],
                            "times" => [0.5,6., 10.]),
                "2" => Dict("points" => [(0.9,0.1), (0.5,0.15), (0.6,0.5), (0.8,0.4), (0.9,0.25)],
                            "times" => [2., 6., 13., 20., 24.]),
                "3" => Dict("points" => [(0.05,0.6), (0.9,0.6)],
                            "times" => [10.,25.]),
                "4" => Dict("points" => [(0.3,0.65), (0.65,0.7), (0.7,0.85), (0.5,0.95), (0.4,0.85)],
                            "times" => [18., 25., 29., 35., 39.]),
                "5" => Dict("points" => [(0.95,0.68), (0.8,0.75), (0.65,0.98)],
                            "times" => [22., 30., 40.]),
                "ncars" => 5)

droneDict = Dict("speed" => 0.05, "startPos" => (0.05,0.05), "goalPos" => (0.95,0.95))

probdict = merge!(Dict("cars"=>carsDict),Dict("drone"=>droneDict))

probdata = JSON.json(probdict)

open("../data/basic_set2.json", "w") do f
  write(f, probdata)
end