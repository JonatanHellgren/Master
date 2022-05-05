policy = reversible_policy

using ColorSchemes, Colors
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTeX-style

cmap = ColorScheme([Colors.RGB(1, 1, 1), Colors.RGB(1, 1, 1), Colors.RGB(1, 1, 1)], "custom", "threetone, red, white, and green")
ColorScheme([get(cmap, i) for i in 0.0:0.001:1.0])

function simulate_mdp(mdp, policy)
  state = initialstate(mdp)
  all_states = [state]
  all_actions = []

  it = 1
  while !(isterminal(mdp, state)) && it < 30
    println(state)
    
    a = action(policy, state)
    println(a)
    append!(all_actions, [a])

    state = transition(mdp, state, a).val
    append!(all_states, [state])

    it += 1
  end

  return all_states, all_actions

end

all_states, all_actions = simulate_mdp(mdp, policy)

function plot_grid_world(state)
    
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

      if state[x, y] == -1
        #= psimage!("@black_arrow.jpg", D="g$x/$y+jCM+w2c", fmt=:jpg, show=1) =#
        color="green"
        rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4)
        plot!(rect, fillalpha=0, linecolor=color)

      elseif state[x, y] == -2
        color="blue"
        rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4)
        plot!(rect, fillalpha=0, linecolor=color)

      elseif state[x, y] == 4
        color="black"
        rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4)
        plot!(rect, fillalpha=0, linecolor=color)

      elseif state[x, y] == 6
        color="gray"
        rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4)
        plot!(rect, fillalpha=0, linecolor=color)

      elseif state[x, y] == 1
        color="yellow"
        rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4)
        plot!(rect, fillalpha=0, linecolor=color)

      end
    end
    
    title!("Reversible")

    return fig
end

anim = @animate for state ∈ all_states[1:end-1]
    plot_grid_world(state)
end
gif(anim, "reversible_2fps.gif", fps = 2)

