using Flux
using Random
using LinearAlgebra
using StatsBase

Xx = Float32.(rand(5,5,4,1))

"""
Function for creating neural net
"""
function init_model(
  params::GridWorldParameters,
  conv_size::Tuple{Int, Int},
  n_conv::Int,
  hidden_dim::Int,
  output_dim::Int
  )

  flat_dim = (params.size[1] - conv_size[1] + 1) * (params.size[2] - conv_size[2] + 1) * n_conv
  model = Chain(
    Conv(conv_size, params.n_foods+1 => n_conv, relu),
    Flux.flatten,
    Dense(flat_dim, hidden_dim),
    Dense(hidden_dim, hidden_dim),
    Dense(hidden_dim, output_dim)
  )
  
  return model
end

"""
Parameters for networks
"""
conv_size = (3, 3)
n_conv = 8
hidden_dim = 64

"""
hyperparameters for training
"""
timesteps_per_batch = 4800
max_timesteps_per_episide = 1600
total_timesteps = 9600
n_updates_per_iteration = 5
clip = 0.2
lr = 1e-4

"""
initialization of actor and critic
"""
actor = init_model(params, conv_size, n_conv, hidden_dim, length(actions(mdp)))
critic = init_model(params, conv_size, n_conv, hidden_dim, 1)

function get_action(actor, obs, greedy=false)
  probs = softmax(actor(Flux.unsqueeze(obs, 4)))

  if greedy
    a_ind = argmax(probs)[1]
  else
    a_ind = sample(weights(probs[:,1]))
  end
    
  action = A[a_ind]
  log_prob = log(probs[a_ind])

  return action, log_prob
end

function compute_rtgs(batch_rews)
  batch_rtgs = []

  for ep_rews in batch_rews[end:-1:1]
    
    discounted_reward = 0

    for rew in ep_rews[end:-1:1]
      discounted_reward = Float32(rew + discounted_reward * params.discount)
      insert!(batch_rtgs, 1, discounted_reward)
    end
  end

  return batch_rtgs
end


function rollout()
  batch_obs = Array{Float32, 4}(undef, params.size[1], params.size[2], params.n_foods+1, 0)
  batch_acts = []
  batch_log_probs = []
  batch_rews = []
  batch_rtgs = []
  batch_lens = []

  t = 0
  rng = MersenneTwister()
  while t < timesteps_per_batch

    # rewards for this episode
    ep_rews = []

    # start at initial state
    obs = initialstate(mdp)[1]
    
    ep_t = 0
    while ep_t < max_timesteps_per_episide
      ep_t += 1
      t += 1
      
      # collect observation
      batch_obs = cat(batch_obs, obs, dims=4)

      action, log_prob = get_action(actor, obs)
      obs, reward = gen(obs, action, rng)

      # collect reward, action and log prob
      append!(ep_rews, [reward])
      append!(batch_acts, [action])
      append!(batch_log_probs, [log_prob])

      # break rollout when terminal state is hit
      if isterminal(mdp, obs)
        break
      end
    end

    # collect batch length and rewards
    append!(batch_lens, [ep_t + 1])
    append!(batch_rews, [ep_rews])
  end

  # compute batch reward to go(s)
  batch_rtgs = compute_rtgs(batch_rews)

  return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
end

function evaluate(critic, batch_obs)
  # V_{ϕ, k} 
  # dim : 1xn_obs
  V = critic(batch_obs)
  return V[:] # as Vector, with length n_obs
end

function get_log_probs(actor, batch_obs, batch_acts)
  out = actor(batch_obs)
  probs = softmax(out)
  log_probs = log.(probs)

  action_inds = action2ind.(batch_acts)

  action_log_probs = [log_probs[a, i] for (i, a) in enumerate(action_inds)]
  return action_log_probs
end

function action2ind(action)
  return argmax(actions(mdp) .== action)
end

function compute_actor_loss(actor, batch_obs, batch_acts, batch_log_probs, Aₖ)
  curr_log_probs = get_log_probs(actor, batch_obs, batch_acts)
  ratios = exp.(curr_log_probs .- batch_log_probs)

  surr1 = ratios .* Aₖ
  surr2 = clamp.(ratios, 1 - clip, 1 + clip) .* Aₖ

  #= println(Aₖ[1:20]) =#

  actor_loss = -mean( min.(surr1, surr2) )

  return actor_loss
end

function compute_critic_loss(critic, x, y) 
  return Flux.mse(critic(x)', y)
end

actor_opt = ADAM(lr)
critic_opt = Descent(lr)

function learn(actor, critic, actor_opt, critic_opt, total_timesteps)
  local actor_loss
  local critic_loss
  local mean_batch_len = []

  for it in range(1, 100)
    t_so_far = 0
    print(it)
    # ALG STEP 2
    while t_so_far < total_timesteps 
      # ALG STEP 3
      batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = rollout()
      println(mean(batch_lens))
      append!(mean_batch_len, mean(batch_lens))

      # calculate V_{ϕ, k}
      V = evaluate(critic, batch_obs)

      # ALG STEP 5
      # calculate advantage
      Aₖ = batch_rtgs .- V

      # normalize advantages
      Aₖ = (Aₖ .- mean(Aₖ)) / max(std(Aₖ), 1e-10) # using max to avoid zero division

      actor_θ = Flux.params(actor)
      critic_θ = Flux.params(critic)

      println()
      for _ in range(1, n_updates_per_iteration)
        actor_gs = gradient(actor_θ) do 
          actor_loss = compute_actor_loss(actor, batch_obs, batch_acts, batch_log_probs, Aₖ)
          return actor_loss
        end
        println("actor")
        println(actor_loss)

        Flux.update!(actor_opt, actor_θ, actor_gs)

        critic_gs = gradient(critic_θ) do 
          critic_loss = compute_critic_loss(critic, batch_obs, batch_rtgs)
          return critic_loss
        end
        #= println("critic") =#
        #= println(critic_loss) =#

        Flux.update!(critic_opt, critic_θ, critic_gs)
      end


      t_so_far += sum(batch_lens)

    end
  end
  return mean_batch_len
end

mean_batch_len = learn(actor, critic, actor_opt, critic_opt, total_timesteps)
