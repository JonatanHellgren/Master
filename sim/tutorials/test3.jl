using POMDPs, POMDPModelTools, POMDPPolicies, QuickPOMDPs, POMDPSimulators
using Parameters, Random
using DiscreteValueIteration

using ColorSchemes, Colors
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTeX-style

cmap = ColorScheme([Colors.RGB(1, 1, 1), Colors.RGB(1, 1, 1), Colors.RGB(1, 1, 1)], "custom", "threetone, red, white, and green")
ColorScheme([get(cmap, i) for i in 0.0:0.001:1.0])

rng = MersenneTwister(1)

@with_kw struct GridWorldParameters
  size::Tuple{Int, Int} = (6, 4)
  discount::Float64     = 0.95
  n_foods::Int          = 1
end
params = GridWorldParameters()

@enum Action UP RIGHT DOWN LEFT
𝒜 = [UP, DOWN, LEFT, RIGHT]

agent = 1
food = -1
function find_type(type::Int, s::Matrix{Int})
  return findall(==(type), s)
end

abstract type GridWorld <: MDP{Matrix{Int}, Action} end

null_state = Int.(zeros(params.size))

begin
  const MOVEMENTS = Dict(
    UP    => CartesianIndex(-1, 0),
    DOWN  => CartesianIndex(1, 0),
    LEFT  => CartesianIndex(0, -1),
    RIGHT => CartesianIndex(0, 1));
end

inbounds(c::CartesianIndex) = 1 ≤ c[1] ≤ params.size[1] && 1 ≤ c[2] ≤ params.size[2]

cords = [[CartesianIndex(x, y) for x in 1:params.size[1], y in 1:params.size[2]]...]
𝒮 = [null_state]
for food_cord in cords, agent_cord in cords
  state = copy(null_state)
  state[food_cord] = -1
  state[agent_cord] = 1
  append!(𝒮, [state])
end


function T(s::Matrix{Int}, a::Action)
  food_cords = find_type(food, s)
  if length(food_cords) != 1
		return Deterministic(null_state)
	end

  food_cord = food_cords[1]
  agent_cord = find_type(agent, s)[1]
  agent_cord_new = agent_cord + MOVEMENTS[a]

  sp = copy(s)
  if inbounds(agent_cord_new)
    sp[agent_cord, 1] = 0
    sp[agent_cord_new, 1] = 1
  end

  return Deterministic(sp)
end


function R(s, a=missing)
  food_cords = find_type(food, s)
  if length(food_cords) != 1
		return 1
  else
    return -0.04
	end
end

# initial states include all states that has a an agent and a food
𝒮_init = []
for s in 𝒮[2:end]  # no null_state
  if sum(s) == 0
    append!(𝒮_init, [s])
  end
end

termination(s::Matrix{Int}) = s == null_state

mdp = QuickMDP(
  GridWorld,
  states = 𝒮,
  actions = 𝒜,
  transition = T,
  reward = R,
  initialstate = 𝒮_init);

solver = ValueIterationSolver(max_iterations=30);
policy = solve(solver, mdp)

function find_states_with(x::Int, y::Int, value::Int, S::Vector{Matrix{Int}})
  ind = [s[x, y] == value for s in S]
  return S[ind]
end

function get_policy_map(S::Vector{Matrix{Int}}, policy)
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

food_x = 4
food_y = 2

S = find_states_with(food_x, food_y, -1, 𝒮)
policy_map = get_policy_map(S, policy)


function plot_grid_world(food_cord::Tuple{Int, Int}, policy_map::Dict{Any, Any})
    
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

      if (x,y) == food_cord
        color="green"
        rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4)
        plot!(rect, fillalpha=0, linecolor=color)
      else
        #= annotate!([(x, y, string((x,y)))]) =#
        annotate!([(x, y, (policy_map[CartesianIndex(x, y)], :center, 12, "Computer Modern"))])
      end

    end
    
    title!("Grid World")

    return fig
end

plot_grid_world((food_x, food_y), policy_map)
