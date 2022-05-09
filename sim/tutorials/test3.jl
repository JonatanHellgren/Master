using POMDPs, POMDPModelTools, POMDPPolicies, QuickPOMDPs, POMDPSimulators
using Parameters, Random
using DiscreteValueIteration

using ColorSchemes, Colors
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTeX-style

cmap = ColorScheme([Colors.RGB(1, 1, 1), Colors.RGB(1, 1, 1), Colors.RGB(1, 1, 1)], "custom", "threetone, red, white, and green")
ColorScheme([get(cmap, i) for i in 0.0:0.001:1.0])

rng = MersenneTwister(1)

@with_kw struct GridWorldParameters
  size::Tuple{Int, Int} = (5, 5)
  discount::Float64     = 0.95
  n_foods::Int          = 1
end
params = GridWorldParameters()

@enum Action UP RIGHT DOWN LEFT
𝒜 = [UP, DOWN, LEFT, RIGHT]

agent = 1
food = -1
food2 = -2
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
#= food2_cord = CartesianIndex(2, 2) =#
for food2_cord in cords, food_cord in cords, agent_cord in cords
#= for food_cord in cords, agent_cord in cords =#
  state = copy(null_state)
  state[food2_cord] = food2
  state[food_cord] = food
  state[agent_cord] = agent
  if !(state ∈ 𝒮)
    append!(𝒮, [copy(state)])
  end
  if food2_cord != food_cord && food2_cord != agent_cord
    state[food2_cord] = 0
    if !(state ∈ 𝒮)
      append!(𝒮, [copy(state)])
    end
  end
end


function T(s::Matrix{Int}, a::Action)
  food_cords = find_type(food, s)
  if length(food_cords) != 1
		return Deterministic(null_state)
	end

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
init_set = Set([food2, food, agent, 0])
for s in 𝒮[2:end]  # no null_state
  if Set(s) == init_set
    append!(𝒮_init, [s])
  end
end

termination(s::Matrix{Int}) = s == null_state

mdp2 = QuickMDP(
  GridWorld,
  states = 𝒮,
  actions = 𝒜,
  transition = T,
  reward = R,
  initialstate = 𝒮_init);

