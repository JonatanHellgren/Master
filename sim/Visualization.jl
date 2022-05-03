solver = ValueIterationSolver(max_iterations=30);
policy = solve(solver, mdp)

using ColorSchemes, Colors
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTeX-style

cmap = ColorScheme([Colors.RGB(1, 1, 1), Colors.RGB(1, 1, 1), Colors.RGB(1, 1, 1)], "custom", "threetone, red, white, and green")
ColorScheme([get(cmap, i) for i in 0.0:0.001:1.0])

function find_states_with(ind::CartesianIndex, value::Int, S::Vector{Any})
  sub_ind = [s[ind] == value for s in S]
  return S[sub_ind]
end

function get_policy_map(S::Vector{Any}, policy)
  # rotated 90 degrees for visualization
  arrows = Dict(UP => "←",
                DOWN => "→",
                LEFT => "↓",
                RIGHT => "↑")
  policy_actions = Dict()
  for s in S
    agent_cord = find_type(agent, s)[1]
    a = arrows[action(policy, s)]
    policy_actions[agent_cord] = a
    #= append!(policy_actions, (agent_cord, a)) =#
  end
  return policy_actions
end


S = find_states_with(food_cord, -1, 𝒮)
S = find_states_with(food2_cord, -2, S)
policy_map = get_policy_map(S, policy)


function plot_grid_world(
  food_cord::CartesianIndex,
  food2_cord::CartesianIndex,
  box_cords::Vector{CartesianIndex{2}},
  policy_map::Dict{Any, Any})
    
    # reshape to grid
    (xmax, ymax) = params.size
    Uxy = reshape(zeros(xmax*ymax), xmax, ymax)

    # plot values (i.e the U matrix)
    fig = heatmap(Uxy',
                  legend=:none,
                  aspect_ratio=:equal,
                  framestyle=:box,
                  tickdirection=:out,
                  color=cmap.colors)
                  
    xlims!(0.5, xmax+0.5)
    ylims!(0.5, ymax+0.5)
    xticks!(1:xmax)
    yticks!(1:ymax)

    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
    for x in 1:xmax, y in 1:ymax
       
      # outline
      rect = rectangle(1, 1, x - 0.5, y - 0.5)
      plot!(rect, fillalpha=0, linecolor=:gray)

      if CartesianIndex(x,y) == food_cord
        color="green"
        rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4)
        plot!(rect, fillalpha=0, linecolor=color)

      elseif CartesianIndex(x, y) == food2_cord
        color="blue"
        rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4)
        plot!(rect, fillalpha=0, linecolor=color)

      elseif CartesianIndex(x, y) in box_cords
        color="black"
        rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4)
        plot!(rect, fillalpha=0, linecolor=color)

      else
        annotate!([(x, y, (policy_map[CartesianIndex(x, y)], :center, 12, "Computer Modern"))])
      end
    end
    
    title!("Grid World")

    return fig
end

plot_grid_world(food_cord, food2_cord, box_cords, policy_map)
