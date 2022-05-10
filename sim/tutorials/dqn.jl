using DeepQLearning
using POMDPs
using Flux
using POMDPModels
using POMDPSimulators
using POMDPPolicies


model = Chain(
  Conv((3,3), 4 => 8, relu),
  #= Conv((3,3), 4 => 4, relu), =#
  Flux.flatten,
  Dense(72, 64),
  Dense(64, 64),
  Dense(64, length(actions(mdp)))
  )

steps = 1e6

exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=steps))
#= exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=steps)) =#

solver = DeepQLearningSolver(qnetwork = model, max_steps=steps, 
                             exploration_policy = exploration,
                             learning_rate=0.0001,log_freq=10000, eval_freq=10000,
                             recurrence=false,double_q=true, dueling=true, prioritized_replay=true)
policy = solve(solver, mdp)

