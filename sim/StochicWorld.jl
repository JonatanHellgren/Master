using POMDPs, POMDPModelTools, POMDPPolicies, QuickPOMDPs, POMDPSimulators
using Parameters, Random

#= using DiscreteValueIteration =#

rng = MersenneTwister(1)

@with_kw struct GridWorldParameters
  size::Tuple{Int, Int} = (5, 5)
  discount::Float64     = 0.95
  n_foods::Int          = 3
end
params = GridWorldParameters()

Base.:+(c1::CartesianIndex{3}, c2::CartesianIndex{2}) = CartesianIndex(c1[1] + c2[1], c1[2] + c2[2], c1[3])

@enum Action UP RIGHT DOWN LEFT NOOP
A = [UP, RIGHT, DOWN, LEFT, NOOP]

"""
function find_type(type::Int, s::Vector{Int, 3})
  return findall(==(type), s[:,:,type])
end
"""

abstract type GridWorld <: MDP{Tuple{Array{Float32, 3}, Int}, Action} end

begin
  const MOVEMENTS = Dict(
    UP    => CartesianIndex(-1, 0),
    DOWN  => CartesianIndex(1, 0),
    LEFT  => CartesianIndex(0, -1),
    RIGHT => CartesianIndex(0, 1),
    NOOP => CartesianIndex(0, 0));
end

begin
  const ACTION_STR = Dict(
    UP    => "LEFT",
    DOWN  => "RIGHT",
    LEFT  => "DOWN",
    RIGHT => "UP",
    NOOP => "NOOP");
end

begin 
  const LABELS = Dict(
    "agent" => 1,
    "food" => 2,
    "food2" => 3,
    "food3" => 4)
end

null_state = (Float32.(zeros(params.size[1], params.size[2], params.n_foods+1)), 0)
null_cord = CartesianIndex(-1, -1)

inbounds(c::CartesianIndex) = 1 ≤ c[1] ≤ params.size[1] && 1 ≤ c[2] ≤ params.size[2] 

function gen(state, a, rng)

  s = state[1]
  rew = state[2]
  agent_cord = findall(==(1), s[:, :, 1])[1]
  sp = copy(s)

  agent_cord_new = agent_cord + MOVEMENTS[a]
  
  r = -0.04
  sp = copy(s)
  if inbounds(agent_cord_new) && a != NOOP
    # move agent
    agent = s[agent_cord, :]
    new_cell = sp[agent_cord_new,:][2:end]
    sp[agent_cord, :] = zeros(4)

    # check if food eaten
    if agent[2:end] == new_cell
      r = 1
      rew += 1

      food_type = findall(==(1), agent)[2]
      if food_type ≤ params.n_foods
        agent[food_type] = 0
        agent[food_type+1] = 1
      end
    end

    sp[agent_cord_new, :] = agent

  else
    # no movement
    agent_cord_new = agent_cord
  end

  for food in 2:params.n_foods+1
    if 1 in sp[:,:,food] && rand(rng) < 0.0 # chance of food moving
      a = rand(rng, A)     # a random direction
      direction = MOVEMENTS[a]
      food_cord = findall(==(1), sp[:,:,food])[1]
      food_cord_new = food_cord + direction
      if inbounds(food_cord_new) && all(sp[food_cord_new, :] .== 0) && food_cord != agent_cord_new
        sp[food_cord, food] = 0
        sp[food_cord_new, food] = 1
      end

    end
  end

  return (sp = (sp, rew), r = r)
end

#= termination(s::Array{Float32, 3}) = sum(s) == 2 =#
termination(s::Tuple{Array{Float32, 3}, Int}) = s[2] > 2

S_init = copy(null_state[1])
S_init[1,1,1] = 1
S_init[1,1,2] = 1
S_init[1,2,2] = 1
S_init[1,3,2] = 1
S_init[1,4,2] = 1
"""
S_init[5,5,2] = 1
S_init[2,3,3] = 1
S_init[1,5,4] = 1
"""
S_init = (S_init, 0)

mdp = QuickMDP(
  GridWorld,
  actions = A,
  gen = gen,
  isterminal = termination,
  initialstate = [S_init]);



