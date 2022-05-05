include("GridWorld.jl")
include("utilityFunctions.jl")

function T_reset(s::Matrix{Int}, a::Action)
  if R_reset(s) == 1
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


S_reset_init = []
init_set = Set([-2, -1, 0, 1, 4, 6])
for s in 𝒮[2:end]  # no null_state
  if Set(s) == init_set
    append!(S_reset_init, [s])
  end
end

function R_reset(s, a=missing)
  if s == S_init
    return 1
  elseif s == null_state
    return 1
  else
    return -0.04
  end
end

termination_reset(s::Matrix{Int}) = s == null_state

mdp_reset = QuickMDP(
  GridWorld,
  states = 𝒮,
  actions = 𝒜,
  transition = T_reset,
  reward = R_reset,
  isterminal = termination_reset,
  initialstate = 𝒮[6]);

solver = ValueIterationSolver(max_iterations=50);
reset_policy = solve(solver, mdp_reset);
