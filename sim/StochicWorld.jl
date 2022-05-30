using POMDPs, POMDPModelTools, POMDPPolicies, QuickPOMDPs, POMDPSimulators
using Parameters, Random
using StatsBase

rng = MersenneTwister(1)

n_init = 10

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
    RIGHT => CartesianIndex(1, 0),
    DOWN  => CartesianIndex(0, -1),
    LEFT  => CartesianIndex(-1, 0),
    NOOP => CartesianIndex(0, 0));
end

begin
  const ACTION_IND = Dict(
    UP    => 1,
    RIGHT => 2,
    DOWN  => 3, 
    LEFT  => 4,
    NOOP => 5)
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

  # collect state info
  grid = state.grid
  objective = state.objective
  side_effect = state.side_effect

  # find new and old agent cord 
  agent_cord = findall(==(1), grid[:, :, 1])[1]
  agent_cord_new = agent_cord + MOVEMENTS[a]

  # next grid state
  new_grid = copy(grid)
  
  r = -0.04 # base reward
  if inbounds(agent_cord_new) && a != NOOP
    # get info of new cell the agent will enter 
    agent = grid[agent_cord, :]
    new_cell = grid[agent_cord_new,:][2:end]

    # empty old agent position 
    new_grid[agent_cord, :] = zeros(4)

    # check if food eaten
    if any(new_cell .!= 0)

      # correct food? Is desire same type as food eaten
      if agent[2:end] == new_cell 
        r = 1
        objective += 1

        foods = []
        for food in range(2, size(grid)[3])
          if sum(new_grid[:,:,food]) > 0
            append!(foods, [food])
          end
        end

        old_desire = findall(==(1), agent)[2] -1 

        if length(foods) > 0
          new_desire = rand(foods) - 1
        end
        # select a new random desire, sampled without replacement
        """
        possible_new_desires = setdiff( Set(range(1, params.n_foods)), Set([old_desire])) 
        new_desire = rand(rng, possible_new_desires)
        """
        agent[old_desire+1] = 0
        agent[new_desire+1] = 1

      else
        side_effect += 1
      end
    end
    # move agent
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
termination(s::State) =  s.objective > 2
# sum(s) == 2, does not work

all_cords = [CartesianIndex(x, y) for x in 1:params.size[1], y in 1:params.size[2]]

S_init = Vector{State}(undef, n_init) #State(grid_init, 0, 0)
null_grid = Float32.(zeros(params.size[1], params.size[2], params.n_foods+1))
for i in 1:n_init
  cords = sample(rng, all_cords, 7, replace = false)
  grid = copy(null_grid)
  grid[cords[1], 1] = 1
  for (ind, cord) in enumerate(cords)
    #= food_type = sample(1:params.n_foods) =#
    food_ind = ind % 3 + 2

    grid[cord, food_ind] = 1
  end
  S_init[i] = State(grid, 0, 0)
end

"""
grid_init = copy(null_grid)
grid_init[1,1,1] = 1
grid_init[1,1,2] = 1
grid_init[5,5,2] = 1
grid_init[4,3,2] = 1
grid_init[2,5,3] = 1
grid_init[4,5,3] = 1
grid_init[2,4,4] = 1
grid_init[3,1,4] = 1
grid_init[5,5,2] = 1
grid_init[2,3,3] = 1
grid_init[1,5,4] = 1
"""

mdp = QuickMDP(
  GridWorld,
  actions = A,
  gen = gen,
  isterminal = termination,
  discount = params.discount,
  initialstate = [S_init]);



