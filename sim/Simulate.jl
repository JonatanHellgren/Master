#= policy = normal_policy =#

using ColorSchemes, Colors
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTeX-style
using Images
using FileIO

@with_kw struct CBColors
  gray::String      = "#999999"
  orange::String    = "#E69F00"
  lightblue::String = "#56B4E9"
  green::String     = "#009E73"
  yellow::String    = "#F0E44 2"
  darkblue::String  = "#0072B2"
  red::String       = "#D55E00"
  pink::String      = "#CC79A7"
end
colors = CBColors()


cmap = ColorScheme([Colors.RGB(1, 1, 1), Colors.RGB(1, 1, 1), Colors.RGB(1, 1, 1)], "custom", "threetone, red, white, and green")
ColorScheme([get(cmap, i) for i in 0.0:0.001:1.0])

function simulate_mdp(mdp, policy)
  state = initialstate(mdp)[1]
  all_states = [state]
  all_actions = []

  it = 1
  rng = MersenneTwister()
  while it < 30
    
    a = action(policy, state)
    append!(all_actions, [a])

    #= state = transition(mdp, state, a).val =#
    state = gen(state, a, rng)[1]
    append!(all_states, [state])

    it += 1
    if (isterminal(mdp, state))
      break
    end
  end

  return all_states, all_actions

end

function simulate()
  batch_acts = []
  #= batch_states = Array{Float32, 4}(undef, params.size[1], params.size[2], params.n_foods+1, 0) =#
  batch_states = []

  rng = MersenneTwister(3)
  state = initialstate(mdp)[1]
  #= batch_states = cat(batch_states, state.grid, dims=4) =#
  append!(batch_states, [state])
  it = 1
  reward = 0
  running = true 
  while running && it < 100
    it += 1
    action, _ = get_action(actor, state.grid, greedy=true)
    append!(batch_acts, [action])
    state, r = gen(state, action, rng)
    reward += r
    #= batch_states = cat(batch_states, state.grid, dims=4) =#
    append!(batch_states, [state])
    if isterminal(mdp, state)
      break
    end
  end
  return batch_acts, batch_states
end

#= all_states, all_actions = simulate_mdp(mdp, policy) =#

#= img_path = "kisspng-arrow-scalable-vector-graphics-clip-art-black-arrow-5aa8e8acc61c23.9217803715210190528115.jpg" =#
#= img = load(img_path) =#
function plot_grid_world(state, action)
    
    # reshape to grid
    (xmax, ymax) = params.size
    Uxy = reshape(zeros(xmax*ymax), xmax, ymax)

    # plot values (i.e the U matrix)
    fig = heatmap(Uxy',
                  legend=:none,
                  aspect_ratio=:equal,
                  size=(1000,1000),
                  #= framestyle=:box, =#
                  tickdirection=:out,
                  color=cmap.colors,
                  tickfont=font(40,"Computer Modern"))
                  
    xlims!(0.5, xmax+0.5)
    ylims!(0.5, ymax+0.5)
    xticks!(1:xmax)
    yticks!(1:ymax)

    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
    circle(r, x, y) = Shape(x .+ cos.(LinRange(0, 2*π, 100)).*r, y .+ sin.(LinRange(0, 2*π, 100)).*r)


    for x in 1:xmax, y in 1:ymax
       
      # outline
      rect = rectangle(1, 1, x - 0.5, y - 0.5)
      plot!(rect, fillalpha=0, linecolor=:gray)

      if state.grid[x, y, 1] == 1
        if state.grid[x, y, 2] == 1
          color = colors.green
        elseif state.grid[x, y, 3] == 1
          color = colors.darkblue
        elseif state.grid[x, y, 4] == 1
          color = colors.red
        else
          color = colors.yellow
        end

        rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4)
        plot!(rect, fillcolor=color, linecolor="black")

      elseif state.grid[x, y, 2] == 1
        color = colors.green
        circ = circle(0.2, x, y)
        plot!(circ, fillcolor=color, linecolor="black")

      elseif state.grid[x, y, 3] == 1
        color = colors.blue
        circ = circle(0.2, x, y)
        plot!(circ, fillcolor=color, linecolor="black")

      elseif state.grid[x, y, 4] == 1
        color = colors.red
        circ = circle(0.2, x, y)
        plot!(circ, fillcolor=color, linecolor="black")

      end
    end
    
    side_effect = state.side_effect
    objective = state.objective
    title!("Action: $action, Objective: $objective, Side effect: $side_effect")
    title!("Simple grid world",titlefont=font(50,"Computer Modern"))

    return fig
end

all_actions, all_states = simulate()
append!(all_actions, [NOOP])
anim = @animate for (state, action) ∈ zip(all_states, all_actions)
  #= state = all_states[it] =#
  #= action = all_actions[it] =#
  plot_grid_world(state, ACTION_STR[action])
end
gif(anim, "2fps.gif", fps = 1)
