# DEBUGGING
function plot_car_route(car_ep_dict::Dict)

    curr_pos = car_ep_dict["pos"]
    car_route_dict = car_ep_dict["route"]

    sorted_route = sort(collect(car_route_dict),by=x->x[1])

    pts_x = [timept[1][1] for (_,timept) in sorted_route]
    pts_y = [timept[1][2] for (_,timept) in sorted_route]

    plot(pts_x, pts_y, markershape = :hexagon, markercolor =:blue)
    plot!([curr_pos[1]], [curr_pos[2]], markershape =:hexagon, markercolor =:red)

end

function plot_drone_and_active_cars_epoch!(epoch_dict::Dict, drone_pos::Point, goal_pos::Point, color_map::Dict)

    car_info_dict = epoch_dict["car-info"]

    # Plot the start and goal
    p = Gadfly.plot(x=[drone_pos.x, goal_pos.x], y=[drone_pos.y, goal_pos.y], 
        shape=[Shape.star1],Geom.point, Theme(background_color=parse(Colorant,"white"),default_color=RGBA(0.,0.,0.,0.35),point_size=12pt),
        xmin=[-1.],xmax=[1.],ymin=[-1.],ymax=[1.])
      
    for (car_id,car_ep_dict) in car_info_dict
        # if rand() < 0.95
        #     continue
        # end
        if car_ep_dict["route"] != nothing
            curr_pos = car_ep_dict["pos"]
            car_route_dict = car_ep_dict["route"]
            sorted_route = sort(collect(car_route_dict),by=x->x[1])

            pts_x = [timept[1][1] for (_,timept) in sorted_route]
            pts_y = [timept[1][2] for (_,timept) in sorted_route]

            if haskey(color_map,car_id)
                car_color = color_map[car_id]
            else
                car_color = RGB(rand(Uniform(0.1,0.9)), rand(Uniform(0.1,0.9)), rand(Uniform(0.1,0.9)))
                color_map[car_id] = car_color
            end

            append!(p.layers, layer(x=pts_x, y=pts_y, Geom.point, shape=[Shape.xcross], Theme(default_color=car_color, point_size=4pt)))
            append!(p.layers, layer(x=[curr_pos[1]], y=[curr_pos[2]], Geom.point, shape=[Shape.square], Theme(default_color=car_color, point_size=11pt)))
        end
    end    

    return p
end