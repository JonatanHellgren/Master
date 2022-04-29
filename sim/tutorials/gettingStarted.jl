using POMDPs, QMDP, POMDPModels, POMDPSimulators

# initialize problem and solver
pomdp = TigerPOMDP()
solver = QMDPSolver()

# compute policy
policy = solve(solver, pomdp)

# ecaluate the policy
belief_updater = updater(policy) # the default QDMP belief updater (discrete Baysian filter)
init_dist = initialstate_distribution(pomdp) # from POMDPModels
hr = HistoryRecorder(max_steps=100) # from POMDPSimulators
hist = simulate(hr, pomdp, policy, belief_updater, init_dist) # run 100 step simulator
println("reward $(discounted_reward(hist))")
