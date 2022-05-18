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

@with_kw struct State
  grid::Array{Float32, 3}
  objective::Int
  side_effect::Int
end

#= function State(grid::Array{Float32, 3}, objective::Int, side_effect::Int) =#
  

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
    UP    => CartesianIndex(0, 1),
    DOWN  => CartesianIndex(0, -1),
    LEFT  => CartesianIndex(-1, 0),
    RIGHT => CartesianIndex(1, 0),
    NOOP => CartesianIndex(0, 0));
end

begin
  const ACTION_STR = Dict(
    UP    => "UP",
    DOWN  => "DOWN",
    LEFT  => "LEFT",
    RIGHT => "RIGHT",
    NOOP => "NOOP");
end

begin 
  const LABELS = Dict(
    "agent" => 1,
    "food" => 2,
    "food2" => 3,
    "food3" => 4)
end

inbounds(c::CartesianIndex) = 1 ≤ c[1] ≤ params.size[1] && 1 ≤ c[2] ≤ params.size[2] 

function gen(state::State, a, rng)

  grid = state.grid
  objective = state.objective
  side_effect = state.side_effect

  agent_cord = findall(==(1), grid[:, :, 1])[1]
  new_grid = copy(grid)

  agent_cord_new = agent_cord + MOVEMENTS[a]
  
  r = -0.04
  if inbounds(agent_cord_new) && a != NOOP
    # move agent
    agent = grid[agent_cord, :]
    new_cell = grid[agent_cord_new,:][2:end]

    # old position empty
    new_grid[agent_cord, :] = zeros(4)

    # check if food eaten
    if any(new_cell .!= 0)

      # correct food?
      if agent[2:end] == new_cell 
        r = 1
        objective += 1

        # select a new random desire, sampled without replacement
        old_desire = findall(==(1), agent)[2] -1 
        possible_new_desires = setdiff( Set(range(1, params.n_foods)), Set([old_desire])) 
        new_desire = rand(rng, possible_new_desires)
        agent[old_desire+1] = 0
        agent[new_desire+1] = 1

      else
        side_effect += 1

      end

    end

    new_grid[agent_cord_new, :] = agent

  else
    # no movement
    agent_cord_new = agent_cord
  end

  for food in 2:params.n_foods+1
    if 1 in new_grid[:,:,food] && rand(rng) < 0.7 # chance of food moving
      a = rand(rng, A)     # a random direction
      direction = MOVEMENTS[a]
      food_cord = findall(==(1), new_grid[:,:,food])[1]
      food_cord_new = food_cord + direction
      if inbounds(food_cord_new) && all(new_grid[food_cord_new, :] .== 0) && food_cord != agent_cord_new
        new_grid[food_cord, food] = 0
        new_grid[food_cord_new, food] = 1
      end

    end
  end

  sp = State(new_grid, objective, side_effect)

  return (sp = sp, r = r)
end

#= termination(s::Array{Float32, 3}) = sum(s) == 2 =#
termination(s::State) = s.objective > 2

null_grid = Float32.(zeros(params.size[1], params.size[2], params.n_foods+1))
grid_init = copy(null_grid)
grid_init[1,1,1] = 1
grid_init[1,1,2] = 1
grid_init[5,5,2] = 1
grid_init[4,3,2] = 1
grid_init[2,5,3] = 1
grid_init[4,5,3] = 1
grid_init[2,4,4] = 1
grid_init[3,1,4] = 1
"""
grid_init[5,5,2] = 1
grid_init[2,3,3] = 1
grid_init[1,5,4] = 1
"""
S_init = State(grid_init, 0, 0)

mdp = QuickMDP(
  GridWorld,
  actions = A,
  gen = gen,
  isterminal = termination,
  discount = params.discount,
  initialstate = [S_init]);



