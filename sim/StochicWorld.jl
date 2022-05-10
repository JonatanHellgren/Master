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

@enum Action UP RIGHT DOWN LEFT NOOP
A = [UP, DOWN, LEFT, RIGHT, NOOP]

"""
function find_type(type::Int, s::Vector{Int, 3})
  return findall(==(type), s[:,:,type])
end
"""

abstract type GridWorld <: MDP{Array{Int, 3}, Action} end

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

null_state = Int.(zeros(params.size[1], params.size[2], params.n_foods+1))
null_cord = CartesianIndex(-1, -1)

inbounds(c::CartesianIndex) = 1 ≤ c[1] ≤ params.size[1] && 1 ≤ c[2] ≤ params.size[2] 

function gen(s, a, rng)

  agent_cord = findall(==(1), s[:, :, 1])[1]
  sp = copy(s)

  agent_cord_new = agent_cord + MOVEMENTS[a]
  
  r = -0.04
  sp = copy(s)
  if inbounds(agent_cord_new) && a != NOOP
    # move agent
    sp[agent_cord, 1] = 0
    sp[agent_cord_new, 1] = 1

    # check if food eaten
    new_cell = sp[agent_cord_new,:][2:end]
    if any(new_cell .!= 0)
      r = 1
      #= println(new_cell) =#
      food_type = findall(==(1), new_cell)[1]

      sp[agent_cord_new, food_type+1] = 0
      #= println(sp[agent_cord_new, :]) =#
      #= println() =#
    end

  else
    # no movement
    agent_cord_new = agent_cord
  end

  for food in 2:params.n_foods+1
    if 1 in sp[:,:,food] && rand(rng) < 0.5 # chance of food moving
      a = rand(A)     # a random direction
      direction = MOVEMENTS[a]
      food_cord = findall(==(1), sp[:,:,food])[1]
      food_cord_new = food_cord + direction
      if inbounds(food_cord_new) && all(sp[food_cord_new, :] .== 0)
        sp[food_cord, food] = 0
        sp[food_cord_new, food] = 1
      end

    end
  end

  return (sp = sp, r = r)
end

termination(s::Array{Int, 3}) = sum(s) == 1

S_init = copy(null_state)
S_init[1,1,1] = 1
S_init[3,3,2] = 1
S_init[2,3,3] = 1
S_init[1,5,4] = 1

mdp = QuickMDP(
  GridWorld,
  actions = A,
  gen = gen,
  isterminal = termination,
  initialstate = [S_init]);



