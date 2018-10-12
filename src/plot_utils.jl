"""
    plot_drone_and_active_cars_epoch(::Dict, ::Point, ::Point, ::Dict, ::plot_params)

Plots the current position of the drone, the desired drone goal position and all current positions of cars
as well as their remaining routes. Uses a map of car IDs to colors so as to maintain consistent colors
for each car once they become active. Optionally takes a dictionary of plotting parameters for 
drone transparency and size, and car size and car route point size.
"""
function plot_drone_and_active_cars_epoch!(epoch_dict::Dict, drone_pos::Point, goal_pos::Point, color_map::Dict, plot_params::Dict=Dict())

    car_info_dict = epoch_dict["car-info"]

    # Extract plotting params with default values otherwise
    drone_alpha = get(plot_params, "drone_alpha", 0.35)
    drone_ptsize = get(plot_params, "drone_ptsize", 11)
    car_route_ptsize = get(plot_params, "car_route_ptsize", 3)
    car_pos_ptsize = get(plot_params, "car_pos_ptsize", 9)

    # Plot the start and goal
    p = Gadfly.plot(x=[drone_pos.x, goal_pos.x], y=[drone_pos.y, goal_pos.y], 
        shape=[Shape.star1],Geom.point, Theme(background_color=parse(Colorant,"white"),default_color=RGBA(0.,0.,0.,drone_alpha),point_size=drone_ptsize*pt),
        xmin=[-1.],xmax=[1.],ymin=[-1.],ymax=[1.],Guide.xticks(ticks=nothing),Guide.xlabel(nothing),Guide.yticks(ticks=nothing),Guide.ylabel(nothing))
      
    for (car_id,car_ep_dict) in car_info_dict
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

            append!(p.layers, layer(x=pts_x, y=pts_y, Geom.point, shape=[Shape.xcross], Theme(default_color=car_color, point_size=car_route_ptsize*pt)))
            append!(p.layers, layer(x=[curr_pos[1]], y=[curr_pos[2]], Geom.point, shape=[Shape.square], Theme(default_color=car_color, point_size=car_pos_ptsize*pt)))
        end
    end    

    return p
end