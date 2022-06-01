using POMDPs, POMDPModelTools, POMDPPolicies, QuickPOMDPs, POMDPSimulators
using Parameters, Random
using StatsBase

rng = MersenneTwister(1)

n_init = 100000
n_test = 100

@with_kw struct GridWorldParameters
  size::Tuple{Int, Int}       = (7, 7)
  obs_length::Tuple{Int, Int} = (2, 2) # distance for each direction
  discount::Float64           = 0.95
  n_foods::Int                = 3
  base_reward::Float32        = -0.04
  food_reward::Float32        = 1
end
params = GridWorldParameters()

@with_kw struct State
  grid::Array{Float32, 3}
  objective::Int
  side_effect::Int
end

@enum Action UP RIGHT DOWN LEFT NOOP
A = [UP, RIGHT, DOWN, LEFT, NOOP]

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

inbounds(c::CartesianIndex) = 1 ≤ c[1] ≤ params.size[1] && 1 ≤ c[2] ≤ params.size[2] 
inbounds(c::Tuple{Int64, Int64}) = 1 ≤ c[1] ≤ params.size[1] && 1 ≤ c[2] ≤ params.size[2] 

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
  
  r = params.base_reward # -0.04
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
        r = params.food_reward # 1
        objective += 1

        foods = []
        for food in range(2, size(grid)[3])
          if sum(new_grid[:,:,food]) > 0
            append!(foods, [food])
          end
        end

        old_desire = findall(==(1), agent)[2] -1 

        if length(foods) > 0
          new_desire = rand(rng, foods) - 1
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
  o = get_observation(new_grid, agent_cord_new, params)

  return (sp = sp, o = o, r = r)
end

function get_observation(grid, agent_cord, params)
  
  obs_dim = params.obs_length .* 2 .+ (1, 1)
  obs = Array{Float32, 3}(undef, obs_dim[1], obs_dim[2], params.n_foods+1)
  for x in range(-params.obs_length[1], params.obs_length[1]),
    y in range(-params.obs_length[2], params.obs_length[2])

    cord = CartesianIndex(Tuple(agent_cord) .+ (x, y))
    obs_ind = CartesianIndex((x, y) .+ params.obs_length .+ (1, 1))
    if inbounds(cord)
      obs[obs_ind, :] = grid[cord, :]
    else
      obs[obs_ind, :] = -ones(params.n_foods + 1)
    end
  end
  return obs  
end

termination(s::State) =  s.objective > 2

all_cords = [CartesianIndex(x, y) for x in 1:params.size[1], y in 1:params.size[2]]

null_grid = Float32.(zeros(params.size[1], params.size[2], params.n_foods+1))

S_init = Vector{NamedTuple{(:sp, :o), Tuple{State, Array{Float32, 3}}}}(undef, n_init) #State(grid_init, 0, 0)
for i in 1:n_init

  n_cords = sample(rng, range(4,10))
  cords = sample(rng, all_cords, n_cords, replace = false)
  grid = copy(null_grid)
  grid[cords[1], 1] = 1
  #= initial_food = sample(rng, 1:params.n_foods) =#
  for (ind, cord) in enumerate(cords)
    food_ind = sample(rng, 1:params.n_foods) + 1
    #= food_ind = (ind + initial_food) % 3 + 2  =#

    grid[cord, food_ind] = 1

  end
  agent_ind = findall(==(1), grid[:,:,1])[1]
  o = get_observation(grid, agent_ind, params)

  S_init[i] = (sp = State(grid, 0, 0), o = o)
end

#= S_test = Vector{State}(undef, n_test) #State(grid_init, 0, 0) =#
S_test = Vector{NamedTuple{(:sp, :o), Tuple{State, Array{Float32, 3}}}}(undef, n_test) #State(grid_init, 0, 0)
for i in 1:n_test
  cords = sample(rng, all_cords, 7, replace = false)
  grid = copy(null_grid)
  grid[cords[1], 1] = 1
  initial_food = sample(rng, 1:params.n_foods)
  for (ind, cord) in enumerate(cords)
    #= food_type = sample(1:params.n_foods) =#
    food_ind = (ind + initial_food) % 3 + 2 


    grid[cord, food_ind] = 1
  end
  agent_ind = findall(==(1), grid[:,:,1])[1]
  o = get_observation(grid, agent_ind, params)

  S_test[i] = (sp = State(grid, 0, 0), o = o)
end

mdp = QuickPOMDP(
  GridWorld,
  actions = A,
  gen = gen,
  # obstype?
  isterminal = termination,
  discount = params.discount,
  initialstate = [S_init]);

