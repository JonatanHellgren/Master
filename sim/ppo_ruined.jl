using Flux
using Random
using LinearAlgebra
using StatsBase
using Formatting
using StaticArrays

include("StochicWorld.jl")

"""
Parameters for networks
"""
conv_size = (3, 3)
n_conv = 16
hidden_dim = 256

"""
hyperparameters for training
"""
timesteps_per_batch = 9600
max_timesteps_per_episide = 500
total_timesteps = 9600
n_updates_per_iteration = 3
n_epochs = 5000
kick_in = 500
clip = 0.2
lr = 3e-4


"""
Function for creating neural net
"""
function init_model(
  params::GridWorldParameters,
  conv_size::Tuple{Int, Int},
  n_conv::Int,
  hidden_dim::Int,
  output_dim::Array{Int}
  )

  flat_dim = (params.size[1] - conv_size[1] + 1) * (params.size[2] - conv_size[2] + 1) * n_conv
  c1 = Conv(conv_size, params.n_foods+1 => n_conv, relu)
  d1 = Dense(flat_dim, hidden_dim, relu)
  d2 = Dense(hidden_dim, hidden_dim, relu)

  actor = Chain(
    c1, Flux.flatten, d1, d2,
    Dense(hidden_dim, output_dim[1]))

  critic = Chain(
    c1, Flux.flatten, d1, d2,
    Dense(hidden_dim, output_dim[2]))
  
  return actor, critic
end


"""
initialization of actor and critic
"""
function get_action(rng, actor, grid; greedy=false)
  probs = softmax(actor(Flux.unsqueeze(grid, 4))) |> cpu

  if greedy
    a_ind = argmax(probs)[1]
  else
    a_ind = sample(rng, weights(probs[:,1]))
  end
    
  action = A[a_ind]
  log_prob = log(probs[a_ind])

  return action, log_prob
end

function compute_rtgs(batch_rews, params)

  batch_rtgs = Float32.(zeros(timesteps_per_batch))
  ind = timesteps_per_batch

  for ep_rews in batch_rews[end:-1:1]
    discounted_reward = 0

    for rew in ep_rews[end:-1:1]
    #= for ri in length(ep_rews):-1:1 =#
      #= rew = ep_rews[ri] =#
      if rew == 1
        discounted_reward = 0
      end
      discounted_reward = Float32(rew + discounted_reward * params.discount)
      batch_rtgs[ind] = discounted_reward
      ind -= 1
    end
  end

  return batch_rtgs
end


function rollout(mdp, actor, critic, critic_ppo)
  batch_grids = Array{Float32, 4}(undef, params.size[1], params.size[2], params.n_foods+1, timesteps_per_batch) |> gpu
  batch_acts = Array{Int}(undef, timesteps_per_batch)
  batch_log_probs = Array{Float32}(undef, timesteps_per_batch)
  batch_rews = []
  batch_rtgs = Float32[]
  batch_lens = Int32[]

  t = 0
  rng = MersenneTwister()
  while t < timesteps_per_batch

    # rewards for this episode
    ep_rews = []

    # start at initial state
    state = sample(initialstate(mdp)[1])
    
    ep_t = 0
    while ep_t < max_timesteps_per_episide && t < timesteps_per_batch
      t += 1
      ep_t += 1
      
      # collect observation
      #= batch_grids = cat(batch_grids, state.grid, dims=4) =#
      grid = state.grid |> gpu
      batch_grids[:,:,:,t] = grid 

      action, log_prob = get_action(rng, actor, grid)
      state, reward = gen(state, action, rng)
      state = state |> gpu

      # collect reward, action and log prob
      batch_acts[t] = ACTION_IND[action]
      batch_log_probs[t] = log_prob
      append!(ep_rews, [reward])
      """
      append!(batch_acts, [action])
      append!(batch_log_probs, [log_prob])
      """

      # break rollout when terminal state is hit
      if isterminal(mdp, state) 
        break
      end
    end

    # collect batch length and rewards
    append!(batch_lens, [ep_t])
    append!(batch_rews, [ep_rews])
  end
  action_inds = Float32.(zeros(5, timesteps_per_batch))

  for (i,a) in enumerate(batch_acts)
    action_inds[a, i] = 1
  end

  # compute batch reward to go(s)
  #= add_auxiliary_reward!(batch_rews, critic_ppo, batch_lens, batch_grids, λ) =#

  batch_rtgs = compute_rtgs(batch_rews, params)
  add_auxiliary_reward!(batch_rtgs, critic_ppo, batch_lens, batch_grids, λ)

  return batch_grids, batch_acts, batch_log_probs, batch_rtgs, batch_lens, action_inds, batch_rews
end

function add_auxiliary_reward!(batch_rtgs, critic, batch_lens, batch_grids, λ)
  aux_rews = get_auxiliary_rewards(critic, batch_grids) |> cpu
  batch_rtgs += λ .* aux_rews

  """
  ind = 1
  for (ep, batch_len) in enumerate(batch_lens)
    batch_rews[ep] += λ .* aux_rews[ind: batch_len + ind - 1]
    ind += batch_len
  end
  """

end

function run_test(mdp, actor)
  lengths = []
  rewards = []
  side_effects = []
  rng = MersenneTwister(1)

  #= for _ in range(1, 10) =#
    #= state = initialstate(mdp)[1][1] =#
  for state in S_test
    it = 1
    reward = 0
    running = true 
    while running && it < 100
      grid = state.grid |> gpu
      it += 1
      action, _ = get_action(rng, actor, grid, greedy=true)
      state, r = gen(state, action, rng)
      #= println(state) =#
      #= println(r) =#
      #= println() =#
      if r > 0
        reward += r
      end

      if isterminal(mdp, state)
        #= print(state.grid[:,:,1]) =#
        break
      end
    end
    #= println(state.objective) =#
    append!(lengths, [it])
    append!(rewards, [reward])
    append!(side_effects, [state.side_effect])

  end

  return mean(lengths), mean(rewards), mean(side_effects)
end


function evaluate(critic, batch_states)
  # V_{ϕ, k} 
  # dim : 1xn_states
  V = critic(batch_states)
  return V[:] # as Vector, with length n_states
end

function get_log_probs(actor, batch_states, batch_acts, action_inds)
  out = actor(batch_states)
  probs = softmax(out)

  #= action_inds = action2ind.(batch_acts) =#
  #= action_prob = [probs[a, i] for (i, a) in enumerate(action_inds)] =#
  action_probs = sum( probs .* action_inds, dims = 1)
  log_probs = log.(action_probs)
  return log_probs'
end

function compute_actor_loss(actor, batch_states, batch_acts, batch_log_probs, Aₖ, action_inds)
  curr_log_probs = get_log_probs(actor, batch_states, batch_acts, action_inds)
  ratios = exp.(curr_log_probs .- batch_log_probs) |> gpu

  surr1 = ratios .* Aₖ
  surr2 = Float32.(clamp.(ratios, 1 - clip, 1 + clip) .* Aₖ)

  #= println(Aₖ[1:20]) =#

  actor_loss = -mean( min.(surr1, surr2) )

  return actor_loss
end

function penalty(critic)
  return sum(abs2, critic.weight) + sum(abs2, critic.bias)
end

function compute_critic_loss(critic, x, y) 
  return Flux.mse(critic(x)', y)# + 0.001 * penalty(critic)
end


function get_auxiliary_rewards(critic_ppo, batch_grids)
  #= aux_rews = []  =#
  m, n, f, _ = size(batch_grids)
  n_aux = f-2 # number of auxiliary tasks
  aux_tasks = Array{Float32, 4}(undef, m, n, f, n_aux*timesteps_per_batch)

  for it in 1:timesteps_per_batch
    aux_tasks[:, :, :, (n_aux*it-1):(n_aux*it)] = get_other_tasks(batch_grids[:,:,:,it], n_aux) 
  end

  aux_tasks = aux_tasks |> gpu
  aux_rews = reshape(critic_ppo(aux_tasks), (n_aux, timesteps_per_batch)) 
  mean_aux_rews = mean(aux_rews, dims=1)' |> gpu
  
  return mean_aux_rews
end


function get_other_tasks(grid, n_aux)
  m, n, f = size(grid)
  other_grids = Array{Float32, 4}(undef, m, n, f, n_aux)
  grid = grid |> cpu

  agent_ind = findall(==(1), grid[:,:,1])
  agent_cell = grid[agent_ind, :]
  current_task = findall(==(1), agent_cell)[2]
  other_task = findall(==(0), agent_cell)
  agent_cell[current_task] = 0
  for (ind, task) in enumerate(other_task)
    other_grids[:,:,:,ind] = copy(grid)
    agent_cell[task] = 1
    other_grids[agent_ind, :, ind] = agent_cell
    agent_cell[task] = 0
    #= append!(other_grids, [grid_copy]) =#
  end
  return other_grids
end

actor_opt = ADAM(lr, (0.9, 0.999))
critic_opt = ADAM(lr, (0.9, 0.999))

function learn(mdp, actor, critic, actor_opt, critic_opt, total_timesteps, λ, critic_ppo)
  local actor_loss
  local critic_loss
  avg_rewards = []
  avg_lengths = []
  avg_side_effects = []

  for it in range(1, n_epochs)
    t_so_far = 0
    println(it)
    # ALG STEP 2
    while t_so_far < total_timesteps 
      # ALG STEP 3
      batch_grids, batch_acts, batch_log_probs, batch_rtgs, batch_lens, action_inds, batch_rews = rollout(mdp, actor, critic, critic_ppo) |> gpu

      # calculate V_{ϕ, k}
      V = evaluate(critic, batch_grids)

      # ALG STEP 5
      Aₖ = batch_rtgs .- V 

      #= if it > kick_in  =#
        #= Aₖ+= λ .* get_auxiliary_rewards(critic, batch_lens, batch_grids) =#
      #= end =#

      # normalize advantages
      Aₖ = (Aₖ .- mean(Aₖ)) / Float32.(max(std(Aₖ), 1e-10)) # using max to avoid zero division

      actor_θ = Flux.params(actor)
      critic_θ = Flux.params(critic)

      for _ in range(1, n_updates_per_iteration)
        actor_gs = gradient(actor_θ) do 
          actor_loss = compute_actor_loss(actor, batch_grids, batch_acts, batch_log_probs, Aₖ, action_inds)
          #= println(actor_loss) =#
          return actor_loss
        end

        Flux.update!(actor_opt, actor_θ, actor_gs)

        critic_gs = gradient(critic_θ) do 
          critic_loss = compute_critic_loss(critic, batch_grids, batch_rtgs)
          return critic_loss
        end

        Flux.update!(critic_opt, critic_θ, critic_gs)
      end


      t_so_far += sum(batch_lens)

    end

    avg_len, avg_rew, avg_side_effect = run_test(mdp, actor)
    printfmt("avg length: {:.2f}, avg reward: {:.2f}, avg side effect: {:2f}\n", avg_len, avg_rew, avg_side_effect)
    println()
    append!(avg_lengths, [avg_len])
    append!(avg_rewards, [avg_rew])
    append!(avg_side_effects, [avg_side_effect])
    #= println(critic(Flux.unsqueeze(grid,4))) =#
    #= println(critic(aux_tasks)) =#
  end
  return avg_lengths, avg_rewards, avg_side_effects
end

λ_stat = []
#= λ = 1.0 =#
for λ in range(2, 1, 2)
  global actor, critic = init_model(params, conv_size, n_conv, hidden_dim, [length(actions(mdp)), 1]) |> gpu
  @time append!(λ_stat, [learn(mdp, actor, critic, actor_opt, critic_opt, total_timesteps, λ, critic_ppo)])
end
