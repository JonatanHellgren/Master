using POMDPs, POMDPModelTools, POMDPPolicies, QuickPOMDPs, POMDPSimulators
using Parameters, Random
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTeX-style
using ColorSchemes, Colors
using StatsBase
using DeepQLearning
using Flux

cmap = ColorScheme([Colors.RGB(1.0, 1.0, 1.0), Colors.RGB(1.0, 1.0, 1.0), Colors.RGB(1.0, 1.0, 1.0)], "custom", "threetone, red, white, and green")
ColorScheme([get(cmap, i) for i in 0.0:0.001:1.0])


struct Cord
  x::Int
  y::Int
end

struct Agent
  cord::Cord
end

struct Food
  cord::Cord
  type::Int
end

struct State
  agent::Agent
  foods::Vector{Food}
end

struct GridState
  grid::Array{Float64, 3}
end

function Base.size(gs::GridState)
  return size(gs.grid)
end

null_cord = Cord(-1, -1)
@with_kw struct GridWorldParameters
  size::Tuple{Int,Int} = (10, 10)
  null_state::State = State(Agent(null_cord), [Food(null_cord, -1)]) 
  p_transition::Real = 0.7
  food_types::Int = 1
end
 
params = GridWorldParameters()

function State2GridState(s::State)
  grid = zeros(params.size[1], params.size[2], params.food_types + 1) 
  a_cord = s.agent.cord
  grid[a_cord.x, a_cord.y, 1] = 1
  for food in s.foods
    grid[food.cord.x, food.cord.y, food.type+1] = 1
  end
  return GridState(grid)
end

function GridState2State(gs::GridState)
  grid = gs.grid
  a_cord = findall(==(1), grid[:,:,1])[1]
  agent = Agent(Cord(a_cord[1], a_cord[2]))

  foods = []
  for ind in 2:size(grid)[3]
    f_cords = findall(==(1), grid[:,:,ind])
    for f_cord in f_cords
      push!(foods, Food(Cord(f_cord[1], f_cord[2]), ind-1))
    end
  end

  return State(agent, foods)
end


Base.:(==)(c1::Cord, c2::Cord) = c1.x == c2.x && c1.y == c2.y
Base.:+(c1::Cord, c2::Cord) = Cord(c1.x + c2.x, c1.y + c2.y)
inbounds(c::Cord) = 1 ≤ c.x ≤ params.size[1] && 1 ≤ c.y ≤ params.size[2]

@enum Action UP RIGHT DOWN LEFT
𝒜 = [UP, RIGHT, DOWN, LEFT]

γ = 0.95

begin
  const MOVEMENTS = Dict(
    UP    => Cord(0,1),
    DOWN  => Cord(0,-1),
    LEFT  => Cord(-1,0),
    RIGHT => Cord(1,0));
end

state = State(Agent(Cord(2,2)), [Food(Cord(2,3), 1)])

function R(s::State)
  for food in s.foods

    if s.agent.cord == food.cord
      return 1
    end

  return -0.04
  end
end

termination(s::State) = s == params.null_state
params = GridWorldParameters()

abstract type GridWorld <: MDP{State, Action} end

function gen(s, a, rng)
  print(size(s))
  state = GridState2State(s)
  print(state)
  #= if len(s.foods) == 0 =#
    #= return Deterministic(params.null_state) =#
  #= end =#

  # move the agent 
  cord⁺ = state.agent.cord + MOVEMENTS[a]
  if inbounds(cord⁺)
    agent⁺ = Agent(cord⁺)
  else
    agent⁺ = state.agent
  end

  # move all foods
  foods⁺ = copy(state.foods)
  for (ind, food) in enumerate(state.foods)
    if rand(rng) < 0 # chance of food moving
      a⁺ = rand(𝒜)     # a random direction
      direction = MOVEMENTS[a⁺]
      food⁺ = Food(food.cord + direction)

      # move the food, unless it is moving towards the agent
      if food⁺ != state.agent.cord && inbounds(food.cord)
        foods⁺[ind] = food⁺
      end
    end
  end

  # calculate the reward
  r = 0.
  for (ind, food) in enumerate(foods⁺)
    if food.cord == agent⁺.cord
      r = 1
      foods⁺[ind] = Food(null_cord)
    else
      r = -0.04
    end
  end

  # return new state and reward
  s⁺ = State2GridState(State(agent⁺, foods⁺))
  return (sp = s⁺, r = r)
end


initialstate = function () 
  ImplicitDistribution() do rng
    𝒞 = [[Cord(x,y) for x=1:params.size[1], y=1:params.size[2]]...]
    C = sample(rng, 𝒞, 6, replace=false)
    agent = Agent(C[1])
    foods = [Food(c, 1) for c in C[2:end]]
    return State2GridState(State(agent, foods))
  end
end

rng = MersenneTwister(1)
mdp = QuickMDP(GridWorld,
  actions = 𝒜,
  discount = γ,
  initialstate = initialstate,
  isterminal = termination,
  gen = gen);


function plot_grid_world(mdp::MDP, step)
    
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

      if Cord(x,y) == step.s.agent.cord
        color="blue"
        rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4)
        plot!(rect, fillalpha=0, linecolor=color)
      end

      for food in step.s.foods
        if Cord(x,y) == food.cord
          color="red"
          rect = rectangle(0.8, 0.8, x - 0.4, y - 0.4)
          plot!(rect, fillalpha=0, linecolor=color)
        end
      end

    end
    
    title!("Grid World")

    return fig
end

π = FunctionPolicy() do s
  return UP
end

model = Chain(Dense(200, 32), Dense(32, length(actions(mdp))))

exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=10000/2))

solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, 
                             exploration_policy = exploration,
                             learning_rate=0.005,log_freq=500,
                             recurrence=false,double_q=true, dueling=true, prioritized_replay=true)

env = convert(AbstractEnv, mdp)
policy = solve(solver, mdp)

#= sim = RolloutSimulator(max_steps=30) =#
#= steps = collect(stepthrough(mdp, policy, max_steps=2)) =#
#= r_tot = simulate(sim, mdp, policy) =#
#= println("Total discounted reward for 1 simulation: $r_tot") =#
#= steps = collect(stepthrough(mdp, π, max_steps=2)) =#
#= plot_grid_world(mdp) =#
