
solver = ReversibleValueIterationSolver(reset_policy, max_iterations=30);
reversible_policy = solve(solver, mdp);

function combine_policy(policy1, policy2)
  new_q = policy1.qmat + policy2.qmat
  policy = []
  util = []
  for i in 1:size(new_q)[1]
    u, p = findmax(new_q[i,:])
    append!(policy, p)
    append!(util, u)
  end
  
  return ValueIterationPolicy(new_q, util, policy, policy1.action_map, true, mdp)
end

#= reversible_policy = combine_policy(policy, reset_policy) =#
    



