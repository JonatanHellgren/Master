using POMDPs, POMDPModelTools, POMDPPolicies, QuickPOMDPs, POMDPSimulators
using StaticArrays
using Parameters, Random
using StatsBase

rng = MersenneTwister(1)

@with_kw struct GridWorldParameters
  size::Tuple{Int, Int} = (10,10)
  discount::Float64     = 0.95
  n_foods::Int          = 1
end
params = GridWorldParameters()


@enum Action UP RIGHT DOWN LEFT
𝒜 = [UP, RIGHT, DOWN, LEFT]

abstract type GridWorld <: MDP{Array{Int}, Action} end

null_state = Int.(zeros(10,10,1))

begin
  const MOVEMENTS = Dict(
    UP    => CartesianIndex(0,1),
    DOWN  => CartesianIndex(0,-1),
    LEFT  => CartesianIndex(-1,0),
    RIGHT => CartesianIndex(1,0));
end

inbounds(c::CartesianIndex) = 1 ≤ c[1] ≤ params.size[1] && 1 ≤ c[2] ≤ params.size[2]

function gen(s, a, rng)
  #= print("a: ") =#
  #= println(a) =#
  agent_cord = findall(==(1), s[:, :, 1])[1]
  #= print("Agent cord: ") =#
  #= println(agent_cord) =#
  sp = copy(s)

  food_cord = findall(==(-1), s[:, :, 1])[1]
  #= print("Food cord: ") =#
  #= println(food_cord) =#
  #= print(food_cord) =#

  #= println(agent_cord) =#
  #= println(food_cord) =#
  #= println(a) =#

  agent_cord_new = agent_cord + MOVEMENTS[a]
  if inbounds(agent_cord_new)
    sp[agent_cord, 1] = 0
    sp[agent_cord_new, 1] = 1
  end

  r = -0.4
  if agent_cord_new == food_cord
    r = 1
    #= println("SUCCESS!!") =#
    #= println() =#
    sp = null_state
  end

  return (sp = sp, r = r)
end

function initialstate(rng)
  ImplicitDistribution() do rng
    𝒞 = [[CartesianIndex(x,y) for x=1:params.size[1], y=1:params.size[2]]...]
    #= 𝒞 = [𝒞[1:44]; 𝒞[46:100]] =#
    C = sample(rng, 𝒞, 2, replace=false)
    grid = Int.(zeros(params.size[1], params.size[2], params.n_foods))
    grid[C[1], 1] = 1
    grid[C[2], 1] = -1
    #= grid[5, 5, 1] = -1 =#
    #= grid[C[1].x, C[2].y, 1] = 1 =#
    return grid
  end
end

"""
policy = FunctionPolicy() do s
  agent_cord = findall(==(1), s[:, :, 1])[1]
  if agent_cord[1] > 5
    return LEFT
  elseif agent_cord[1] < 5
    return RIGHT
  elseif agent_cord[2] > 5
    return DOWN
  elseif agent_cord[2] < 5
    return UP
  end
end
"""

policy = FunctionPolicy() do s
  return DOWN
end

function termination(s)
  #= agent_cord = findall(==(1), s[:, :, 1])[1] =#
  #= food_cord = findall(==(-1), s[:, :, 1])[1] =#
  return s == null_state
end


mdp = QuickMDP(GridWorld,
  actions = 𝒜,
  gen = gen,
  discount = params.discount,
  initialstate = initialstate(rng),
  isterminal = termination)

#= steps = collect(stepthrough(mdp, policy, max_steps=10)); =#
for (s, sp, a, r) in stepthrough(mdp, policy, "s,sp,a,r", max_steps=1000)
    agent_cord = findall(==(1), s[:, :, 1])[1]
    print("Agent cord: ")
    print(agent_cord)
    print(" -> ")
    if sp ≠ null_state
      agent_cord_p = findall(==(1), sp[:, :, 1])[1]
      println(agent_cord_p)
    else
      println(sp)
    end

    food_cord = findall(==(-1), s[:, :, 1])[1]
    print("Food cord: ")
    println(food_cord)
    @show a
    @show r
    println()
end
