#= policy = normal_policy =#

using ColorSchemes, Colors
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTeX-style
using Images
using FileIO

cmap = ColorScheme([Colors.RGB(1, 1, 1), Colors.RGB(1, 1, 1), Colors.RGB(1, 1, 1)], "custom", "threetone, red, white, and green")
ColorScheme([get(cmap, i) for i in 0.0:0.001:1.0])

function simulate_mdp(mdp, policy)
  state = initialstate(mdp)[1]
  all_states = [state]
  all_actions = []

  it = 1
  rng = MersenneTwister()
  while   it < 30
    
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

all_states, all_actions = simulate_mdp(mdp, policy)

#= img_path = "kisspng-arrow-scalable-vector-graphics-clip-art-black-arrow-5aa8e8acc61c23.9217803715210190528115.jpg" =#
#= img = load(img_path) =#
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
    circle(r, x, y) = Shape(x .+ cos.(LinRange(0, 2*π, 100)).*r, y .+ sin.(LinRange(0, 2*π, 100)).*r)


    for x in 1:xmax, y in 1:ymax
       
      # outline
      rect = rectangle(1, 1, x - 0.5, y - 0.5)
      plot!(rect, fillalpha=0, linecolor=:gray)

      if state[x, y] == -1
        #= psimage!("@black_arrow.jpg", D="g$x/$y+jCM+w2c", fmt=:jpg, show=1) =#
        #= plot!(img, inset = bbox(x, y, x+1, y+1, :center),) =#
        color="green"
        #= rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4) =#
        #= plot!(rect, fillalpha=0, linecolor=color) =#
        circ = circle(0.2, x, y)
        plot!(circ, fillcolor=color, linecolor=color)

      elseif state[x, y] == -2
        color="blue"
        #= rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4) =#
        #= plot!(rect, fillalpha=0, linecolor=color) =#
        #= circ = (1, x, y) =#
        #= plot!(circ, fillalpha=1, linecolor=color) =#
        circ = circle(0.2, x, y)
        plot!(circ, fillcolor=color, linecolor=color)

      elseif state[x, y] == -3
        color="red"
        #= rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4) =#
        #= plot!(rect, fillalpha=0, linecolor=color) =#
        circ = circle(0.2, x, y)
        plot!(circ, fillcolor=color, linecolor=color)

      elseif state[x, y] == 4
        color="black"
        rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4)
        plot!(rect, fillcolor=color, linecolor=color)

      elseif state[x, y] == 6
        color="gray"
        rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4)
        plot!(rect, fillalpha=0, linecolor=color)

      elseif state[x, y] == 1
        color="yellow"
        rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4)
        plot!(rect, fillcolor=color, linecolor=color)

      end
    end
    
    title!("Reversible")

    return fig
end

anim = @animate for state ∈ all_states
    plot_grid_world(state)
end
gif(anim, "2fps.gif", fps = 2)

