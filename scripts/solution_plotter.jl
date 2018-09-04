using JSON
using HitchhikingDrones
using Gadfly

# Arguments are <soln filename> <prefix-of-solnvid>
soln_fn = ARGS[1]
outfile_pref = ARGS[2]

ep_dict = Dict()
open(soln_fn,"r") do f
    global ep_dict
    ep_dict = JSON.parse(f)    
end

num_epochs = ep_dict["num_epochs"]

goal_pos = Point(ep_dict["goal_pos"][1], ep_dict["goal_pos"][2])
color_map = Dict()

for epoch = 1:num_epochs

    println("Plotting EPOCH ",epoch)

    img_fn = string(outfile_pref,"-",epoch,".png")
    epoch_dict = ep_dict["epochs"][string(epoch)]
    curr_pos = Point(epoch_dict["drone-info"]["pos"][1], epoch_dict["drone-info"]["pos"][2])
    p = plot_drone_and_active_cars_epoch!(epoch_dict, curr_pos, goal_pos, color_map)
    draw(PNG(img_fn, 10inch, 10inch), p)
end