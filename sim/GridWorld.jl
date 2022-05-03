using POMDPs, POMDPModelTools, POMDPPolicies, QuickPOMDPs, POMDPSimulators
using Parameters, Random

using DiscreteValueIteration

rng = MersenneTwister(1)

@with_kw struct GridWorldParameters
  size::Tuple{Int, Int} = (5, 4)
  discount::Float64     = 0.95
  n_foods::Int          = 1
end
params = GridWorldParameters()

@enum Action UP RIGHT DOWN LEFT
𝒜 = [UP, DOWN, LEFT, RIGHT]


function find_type(type::Int, s::Matrix{Int})
  return findall(==(type), s)
end

abstract type GridWorld <: MDP{Matrix{Int}, Action} end

begin
  const MOVEMENTS = Dict(
    UP    => CartesianIndex(-1, 0),
    DOWN  => CartesianIndex(1, 0),
    LEFT  => CartesianIndex(0, -1),
    RIGHT => CartesianIndex(0, 1));
end

agent = 1
food = -1
food2 = -2
box = 4

agent_cord = CartesianIndex(5, 3)

food_cord = CartesianIndex(1, 3)
food2_cord = CartesianIndex(3, 3)
food_cords = [food_cord, food2_cord]

box1_cord = CartesianIndex(3, 4)
box2_cord = CartesianIndex(3, 2)
box_cords = [box1_cord, box2_cord]

null_state = Int.(zeros(params.size))
null_state[box1_cord] = box
null_state[box2_cord] = box

cords = [[CartesianIndex(x, y) for x in 1:params.size[1], y in 1:params.size[2]]...]
𝒮 = [null_state]
for agent_cord in cords
  if !(agent_cord in box_cords)
    state = copy(null_state)
    state[food_cord] = food
    state[food2_cord] = food2
    state[agent_cord] = agent
    if !(state ∈ 𝒮)
      append!(𝒮, [copy(state)])
    end

    if agent_cord != food_cord
      state_copy = copy(state)
      state_copy[food_cord] = 0
      if !(state_copy ∈ 𝒮)
        append!(𝒮, [copy(state_copy)])
      end
    end

    if agent_cord != food2_cord
      state_copy = copy(state)
      state_copy[food2_cord] = 0
      if !(state_copy ∈ 𝒮)
        append!(𝒮, [copy(state_copy)])
      end
    end

    if agent_cord != food2_cord && agent_cord != food_cord
      state_copy = copy(state)
      state_copy[food_cord] = 0
      state_copy[food2_cord] = 0
      if !(state_copy ∈ 𝒮)
        append!(𝒮, [copy(state_copy)])
      end
    end

  end
end


inbounds(c::CartesianIndex) = 1 ≤ c[1] ≤ params.size[1] && 1 ≤ c[2] ≤ params.size[2] && !(c in box_cords)

function T(s::Matrix{Int}, a::Action)
  if R(s) == 1
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



# initial states include all states that has a an agent and a food
"""
𝒮_init = []
init_set = Set([food2, food, agent, 0])
for s in 𝒮[2:end]  # no null_state
  if Set(s) == init_set
    append!(𝒮_init, [s])
  end
end
"""
S_init = copy(null_state)
S_init[food_cord] = food
S_init[food2_cord] = food2
S_init[agent_cord] = agent

S_reset_init = copy(S_init)
S_reset_init[agent_cord] = 0
S_reset_init[1,1] = 1

function R(s, a=missing)
  food_cords = find_type(food, s)
  if length(food_cords) != 1
		return 1
  else
    return -0.04
	end
end

termination(s::Matrix{Int}) = s == null_state

mdp = QuickMDP(
  GridWorld,
  states = 𝒮,
  actions = 𝒜,
  transition = T,
  reward = R,
  isterminal = termination,
  initialstate = S_init);

solver = ValueIterationSolver(max_iterations=30);
normal_policy = solve(solver, mdp);
