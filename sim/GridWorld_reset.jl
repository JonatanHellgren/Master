include("GridWorld.jl")

𝒮_reset = [null_state]
for agent_cord in cords
  if !(agent_cord in box_cords)
    state = copy(null_state)
    state[food_cord] = food
    state[food2_cord] = food2
    state[agent_cord] = agent
    if !(state ∈ 𝒮_reset)
      append!(𝒮_reset, [copy(state)])
    end

    if agent_cord != food_cord
      state_copy = copy(state)
      state_copy[food_cord] = 0
      if !(state_copy ∈ 𝒮_reset)
        append!(𝒮_reset, [copy(state_copy)])
      end
    end

    if agent_cord != food2_cord
      state_copy = copy(state)
      state_copy[food2_cord] = 0
      if !(state_copy ∈ 𝒮_reset)
        append!(𝒮_reset, [copy(state_copy)])
      end
    end

    if agent_cord != food2_cord && agent_cord != food_cord
      state_copy = copy(state)
      state_copy[food_cord] = 0
      state_copy[food2_cord] = 0
      if !(state_copy ∈ 𝒮_reset)
        append!(𝒮_reset, [copy(state_copy)])
      end
    end

  end
end

function T_reset(s::Matrix{Int}, a::Action)
  if R_reset(s) == 1
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


S_reset_init = copy(S_init)
S_reset_init[agent_cord] = 0
S_reset_init[1,1] = 1

function R_reset(s, a=missing)
  if s == S_init
    return 1
  elseif s == null_state
    return 1
  else
    return -0.04
  end
end

termination(s::Matrix{Int}) = s == null_state

mdp_reset = QuickMDP(
  GridWorld,
  states = 𝒮,
  actions = 𝒜,
  transition = T_reset,
  reward = R_reset,
  isterminal = termination,
  initialstate = S_reset_init);

solver = ValueIterationSolver(max_iterations=30);
reset_policy = solve(solver, mdp_reset);
