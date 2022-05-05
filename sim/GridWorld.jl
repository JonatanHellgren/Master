using POMDPs, POMDPModelTools, POMDPPolicies, QuickPOMDPs, POMDPSimulators
using Parameters, Random

using DiscreteValueIteration

rng = MersenneTwister(1)

@with_kw struct GridWorldParameters
  size::Tuple{Int, Int} = (7, 4)
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

begin 
  const LABELS = Dict(
    "agent" => 1,
    "food" => -1,
    "food2" => -2,
    "box" => 4,
    "mbox" => 6); #movable box
end

begin 
  const INITIAL_CORDS = Dict(
    "agent_cord" => CartesianIndex(7, 3),
    "food_cord" => CartesianIndex(1, 3),
    "food2_cord" => CartesianIndex(4, 3),
    #= "food_cords" => [, food2_cord], =#
    "box1_cord" => CartesianIndex(4, 4),
    "box2_cord" => CartesianIndex(4, 2),
    "box_cords" => [CartesianIndex(4, 4), CartesianIndex(4, 2)],
    "mbox_cord" => CartesianIndex(4, 1));
end


null_state = Int.(zeros(params.size))
null_state[INITIAL_CORDS["box1_cord"]] = LABELS["box"]
null_state[INITIAL_CORDS["box2_cord"]] = LABELS["box"]

function get_states(params)

  cords = [[CartesianIndex(x, y) for x in 1:params.size[1], y in 1:params.size[2]]...]
  cords = subtract_cords(cords, INITIAL_CORDS["box_cords"])

  𝒮 = [null_state]
  for a_cord in cords, m_cord in cords
    state = copy(null_state)
    state[INITIAL_CORDS["food_cord"]] = LABELS["food"]
    state[INITIAL_CORDS["food2_cord"]] = LABELS["food2"]
    state[m_cord] = LABELS["mbox"]
    state[a_cord] = LABELS["agent"]

    append_if_possible(state, 𝒮)

    append_all_combinations([INITIAL_CORDS["food_cord"], INITIAL_CORDS["food2_cord"]], a_cord, state, 𝒮)
  end

  return 𝒮
end

function subtract_cords(all_cords, subtracting_cords)
  filter(x -> !(x in subtracting_cords), all_cords)
end

function append_if_possible(state, 𝒮)
  if !(state ∈ 𝒮)
    append!(𝒮, [copy(state)])
  end
end

function append_all_combinations(changeable_cords, a_cord, state, 𝒮)
  state1 = copy(state)
  for cord in changeable_cords
    state2 = copy(state)
    if cord != a_cord
      state1[cord] = 0
      state2[cord] = 0

      append_if_possible(state2, 𝒮)
    end
  end

  append_if_possible(state1, 𝒮)
end

𝒮 = get_states(params)





"""
for agent_cord in cords, mbox_cord in cords
  if !(agent_cord in box_cords)
    state = copy(null_state)
    state[food_cord] = food
    state[food2_cord] = food2
    state[agent_cord] = agent

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
"""

inbounds(c::CartesianIndex) = 1 ≤ c[1] ≤ params.size[1] && 1 ≤ c[2] ≤ params.size[2] && !(c in INITIAL_CORDS["box_cords"])

function T(s::Matrix{Int}, a::Action)
  if R(s) == 1
		return Deterministic(null_state)
	end

  agent_cord = find_type(LABELS["agent"], s)[1]
  agent_cord_new = agent_cord + MOVEMENTS[a]

  sp = copy(s)
  if inbounds(agent_cord_new)
    
    # move box if movable
    if sp[agent_cord_new] == LABELS["mbox"]
      new_box_cord = agent_cord_new + MOVEMENTS[a]
      if inbounds(new_box_cord)
        sp[agent_cord] = 0
        sp[agent_cord_new] = 1
        sp[new_box_cord] = LABELS["mbox"]
      end
    end

    sp[agent_cord] = 0
    sp[agent_cord_new] = 1
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
S_init = copy(𝒮[1])
S_init[INITIAL_CORDS["food_cord"]] = LABELS["food"]
S_init[INITIAL_CORDS["food2_cord"]] = LABELS["food2"]
S_init[INITIAL_CORDS["agent_cord"]] = LABELS["agent"]
S_init[INITIAL_CORDS["mbox_cord"]] = LABELS["mbox"]

function R(s, a=missing)
  food_cords = find_type(LABELS["food"], s)
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
