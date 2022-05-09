using DeepQLearning
using POMDPs
using Flux
using POMDPModels
using POMDPSimulators
using POMDPPolicies


model = Chain(
  Conv((3,3), 1 => 4, relu),
  #= Conv((3,3), 4 => 4, relu), =#
  Flux.flatten,
  Dense(36, 64),
  Dense(64, 64),
  Dense(64, length(actions(mdp)))
  )

steps = 1e7

exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=steps))
#= exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=steps)) =#

solver = DeepQLearningSolver(qnetwork = model, max_steps=steps, 
                             exploration_policy = exploration,
                             learning_rate=0.0001,log_freq=10000, eval_freq=10000,
                             recurrence=false,double_q=true, dueling=true, prioritized_replay=true)
policy = solve(solver, mdp)

#= sim = RolloutSimulator(max_steps=50) =#
for (s, sp, a, r) in stepthrough(mdp, policy, "s,sp,a,r", max_steps=10)
    agent_cord = findall(==(1), s)[1]
    print("Agent cord: ")
    print(agent_cord)

    #= food_cord = findall(==(-1), s[:, :, 1])[1] =#
    #= print("Food cord: ") =#
    #= println(food_cord) =#
    @show a
    @show r
    println()
end
#= r_tot = simulate(sim, mdp, policy) =#
#= println("Total discounted reward for 1 simulation: $r_tot") =#
