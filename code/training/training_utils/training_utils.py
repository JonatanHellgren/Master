import numpy as np
from torch import tensor, long, float

    # Move out, rollout
    # Make usable for manager
    # Add aux rewards in here

def rollout(agent, train_parameters, mdp, use_aux=False):
    # Batch data
    batch_obs = []
    batch_acts = []
    batch_log_probs = []
    batch_rews = []
    batch_rtgs = []
    batch_lens = []

    time_step = 0

    while time_step < train_parameters.timesteps_per_batch:
        # rewards from episode
        ep_rews = []
        obs = mdp.reset()
        done = False
        for ep_time_step in range(train_parameters.max_timesteps_per_episode):
            time_step += 1

            # Collect observation
            batch_obs.append(np.copy(obs))

            action, log_prob = agent.get_action(obs)
            obs, rew, done, _ = mdp.step(int(action))

            # Collect reward, action, and log prob
            ep_rews.append(rew)
            batch_acts.append(action)
            batch_log_probs.append(log_prob)

            if done:
                break

        # Collect episodic length and rewards
        batch_lens.append(ep_time_step + 1) # +1, since t is initalized as 0
        batch_rews.append(ep_rews)

    # Convert batch data to tensors
    batch_obs = tensor(np.array(batch_obs), dtype=float)
    batch_acts = tensor(batch_acts, dtype=long)
    batch_log_probs = tensor(batch_log_probs, dtype=float)

    batch_rtgs = _compute_rtgs(batch_rews, train_parameters)

    return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

def _compute_rtgs(batch_rews, train_parameters):
    batch_rtgs = []

    for ep_rews in reversed(batch_rews):
        discounted_reward = 0

        for rew in reversed(ep_rews):
            discounted_reward = rew + discounted_reward * train_parameters.gamma
            batch_rtgs.insert(0, discounted_reward)

    batch_rtgs = tensor(batch_rtgs, dtype=float)

    return batch_rtgs
