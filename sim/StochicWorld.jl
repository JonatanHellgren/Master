using POMDPs, POMDPModelTools, POMDPPolicies, QuickPOMDPs, POMDPSimulators
using Parameters, Random

using DiscreteValueIteration

rng = MersenneTwister(1)

@with_kw struct GridWorldParameters
  size::Tuple{Int, Int} = (5, 5)
  discount::Float64     = 0.95
  n_foods::Int          = 3
end
params = GridWorldParameters()

Base.:+(c1::CartesianIndex{3}, c2::CartesianIndex{2}) = CartesianIndex(c1[1] + c2[1], c1[2] + c2[2], c1[3])

@enum Action UP RIGHT DOWN LEFT
A = [UP, DOWN, LEFT, RIGHT]

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

begin 
  const LABELS = Dict(
    "agent" => 1,
    "food" => -1,
    "food2" => -2,
    "food3" => -3,
    "box" => 4,
    "mbox" => 6); #movable box
end

null_state = Int.(zeros(params.size[1], params.size[2], 1))
null_cord = CartesianIndex(-1, -1)

function get_states(params)

  cords = [[CartesianIndex(x, y) for x in 1:params.size[1], y in 1:params.size[2]]...]
  S = [null_state]
  for a_cord in cords
    state = copy(null_state)
    state[a_cord] = LABELS["agent"]
    append_if_possible(state, S)
    cord_subset = subtract_cords(cords, [a_cord])

    all_food_cords = all_combinations(cord_subset, null_cord)
    for food_cords in all_food_cords
      state_copy = copy(state)
      for (type, cord) in enumerate(food_cords)
        if cord != null_cord
          state_copy[cord] = -type
        end
      end
      append_if_possible(state_copy, S)
    end

  end

  #= S = vec([(state, desire) for state in S, desire in [-1, -2, -3]]) =#
  return S
end
# 137960 for (3, 3) grid

function subtract_cords(all_cords, subtracting_cords)
  filter(x -> !(x in subtracting_cords), all_cords)
end

function append_if_possible(state, S)
  if !(state ∈ S)
    append!(S, [copy(state)])
  end
end

function all_combinations(cord_subset::Vector{T}, null_cord) where T
  combinations = Vector{T}[]
  append!(cord_subset, [CartesianIndex(-1, -1)])

  for cord in cord_subset, cord2 in cord_subset, cord3 in cord_subset
    comb = [cord, cord2, cord3]
    n_null_cords = length(filter(x -> (x == null_cord), comb))
    if length(Set(comb)) == 3 || n_null_cords == 2 || n_null_cords == 3
      if !(comb ∈ combinations)
        append!(combinations, [comb])
      end
    end
  end

  return combinations
end

inbounds(c::CartesianIndex) = 1 ≤ c[1] ≤ params.size[1] && 1 ≤ c[2] ≤ params.size[2] 

function R(s, a)

  if s == null_state
    return 1
  end

  agent_cord = find_type(LABELS["agent"], s)[1]
  agent_cord_new = agent_cord + MOVEMENTS[a]
  
  if inbounds(agent_cord_new) && s[agent_cord_new] != 0
    return 1
  else
    return -0.04
  end
end

function T(s::Matrix{Int}, a::Action)
  if sum(s) == 1 || sum(s) == 0
		return Deterministic(null_state)
	end

  agent_cord = find_type(LABELS["agent"], s)[1]
  agent_cord_new = agent_cord + MOVEMENTS[a]

  sp = copy(s)
  if inbounds(agent_cord_new)

    sp[agent_cord] = 0
    sp[agent_cord_new] = 1
  end

  return Deterministic(sp)
end

function gen(s, a, rng)

  agent_cord = findall(==(1), s[:, :, 1])[1]
  sp = copy(s)

  agent_cord_new = agent_cord + MOVEMENTS[a]
  
  r = -0.4
  sp = copy(s)
  if inbounds(agent_cord_new)
    sp[agent_cord, 1] = 0
    sp[agent_cord_new, 1] = 1

    if s[agent_cord_new] != 0
      r = 1
    end

  else
    agent_cord_new = agent_cord
  end

  for food in [-1, -2, -3]
    if food in sp rand(rng) < 0.5 # chance of food moving
      a = rand(A)     # a random direction
      direction = MOVEMENTS[a]
      food_cord = findall(==(food), sp)[1]
      food_cord_new = food_cord + direction
      if inbounds(food_cord_new) && sp[food_cord_new] == 0
        sp[food_cord] = 0
        sp[food_cord_new] = food
      end

    end
  end
  """
  if sum(sp) == 1 || sum(sp) == 0
    return (sp = null_state, r = 0)
	end
  """

  return (sp = sp, r = r)
end

#= S = get_states(params) =#
#= termination(s::Matrix{Int}) = s == null_state =#
termination(s::Array{Int, 3}) = sum(s) == 1

S_init = copy(null_state)
S_init[1,1] = 1
S_init[3,3] = -1
S_init[2,3] = -2
S_init[1,5] = -3

mdp = QuickMDP(
  GridWorld,
  #= states = S, =#
  actions = A,
  gen = gen,
  #= transition = T, =#
  #= reward = R, =#
  isterminal = termination,
  initialstate = [S_init]);

#= solver = ValueIterationSolver(max_iterations=30); =#
#= normal_policy = solve(solver, mdp); =#


