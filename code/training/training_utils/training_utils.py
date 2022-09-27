from collections import defaultdict

import numpy as np
import torch

def rollout_test_set(agent, train_parameters, mdp):
    """
    Lets the agent go through the entire test distribution with the greedy policy.
    Return stats for logging.
    """
    # Batch data
    data = defaultdict(list)

    dones = 0
    for test in mdp.test_set:
        # rewards from episode
        obs = mdp.set_initial_state(np.copy(test))
        ep_time_step, ep_rews, done = \
                _ep_rollout(mdp, obs, train_parameters, data, agent, greedy=True)
        if done:
            dones += 1

        # Collect episodic length and rewards
        data["batch_lens"].append(ep_time_step + 1) # +1, since t is initalized as 0
        data["batch_rews"].append(ep_rews)
        data["objectives"].append(mdp.objectives)
        data["side_effects"].append(mdp.side_effects)

    # computing avareges and rounding to two decimals
    avg_len = round(np.mean(data["batch_lens"]), 2)
    avg_obj = round(np.mean(data["objectives"]), 2)
    avg_side_effects = round(np.mean(data["side_effects"]), 2)

    batch_obs = torch.tensor(np.array(data["batch_obs"]), dtype=torch.float)
    batch_rtgs = _compute_rtgs(data["batch_rews"], train_parameters)

    return batch_obs, batch_rtgs, avg_len, avg_obj, avg_side_effects, dones, data["batch_lens"]

def rollout(agent, train_parameters, mdp, lmbda):
    """
    Function to fill a batch with rollouts.
    Returns all information from the rollout.
    """
    data = defaultdict(list)

    time_step = 1
    while time_step < train_parameters.timesteps_per_batch:
        # rewards from episode
        obs = mdp.reset()
        ep_time_step, ep_rews, _ = _ep_rollout(mdp, obs, train_parameters, data, agent)
        time_step += ep_time_step

        # Collect episodic length and rewards
        data["batch_lens"].append(ep_time_step + 1) # +1, since t is initalized as 0
        data["batch_rews"].append(ep_rews)

    # Convert batch data to tensors
    batch_obs = torch.tensor(np.array(data["batch_obs"]), dtype=torch.float)
    batch_acts = torch.tensor(data["batch_acts"], dtype=torch.long)
    batch_log_probs = torch.tensor(data["batch_log_probs"], dtype=torch.float)

    # if lambda exsits, then compute the auxilary reward using it
    if lmbda is not None:
        data["batch_rews"] = \
                _add_auxiliary_reward(batch_obs, data["batch_rews"], mdp.pomdp,
                                      agent.manager, lmbda)

    # compute rtgs
    batch_rtgs = _compute_rtgs(data["batch_rews"], train_parameters)

    return batch_obs, batch_acts, batch_log_probs, batch_rtgs, data["batch_lens"]

def _add_auxiliary_reward(batch_obs, batch_rews, pomdp, manager, lmbda):
    idx = 0
    for i, ep_rews in enumerate(batch_rews):
        batch_len = len(ep_rews)

        # get auxiliary tasks from batch
        auxiliary_tasks_1, auxiliary_tasks_2 = _get_auxiliary_tasks(
                batch_obs[idx:(idx+batch_len)], pomdp)

        # compute managers estimate for future rewards in the auxiliary tasks
        aux_rews_1 = manager(auxiliary_tasks_1).detach()
        aux_rews_2 = manager(auxiliary_tasks_2).detach()

        # summarize the reward
        aux_rews = torch.cat([aux_rews_1, aux_rews_2], 1)
        sum_aux_rews = torch.sum(aux_rews, dim=1)

        # and compute the relative change in each state
        relative_aux_reward = sum_aux_rews[1:] - sum_aux_rews[0:-1]

        # adding the auxiliary reward to the batch reward
        ep_rews_tensor = torch.tensor(ep_rews, dtype=torch.float).to(manager.device)
        ep_rews_tensor[0:-1] += lmbda * relative_aux_reward
        batch_rews[i] = [float(r) for r in ep_rews_tensor]

        idx += batch_len
    return batch_rews


def _ep_rollout(mdp, obs, train_parameters, data, agent, greedy=False):
    """
    Rollouts a single episode
    """
    ep_rews = []
    done = False
    for ep_time_step in range(train_parameters.max_timesteps_per_episode):

        # Collect observation
        data["batch_obs"].append(np.copy(obs))

        action, log_prob = agent.get_action(obs, greedy)
        obs, rew, done, _ = mdp.step(int(action))

        # Collect reward, action, and log prob
        ep_rews.append(rew)
        data["batch_acts"].append(action)
        data["batch_log_probs"].append(log_prob)

        if done:
            break

    return ep_time_step, ep_rews, done

def _compute_rtgs(batch_rews, train_parameters):
    """
    computes the reward-to-go
    """
    batch_rtgs = []
    # print(batch_rews)

    for ep_rews in reversed(batch_rews):
        discounted_reward = 0

        for rew in reversed(ep_rews):
            discounted_reward = rew + discounted_reward * train_parameters.gamma
            batch_rtgs.insert(0, discounted_reward)

    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

    return batch_rtgs

def _get_auxiliary_tasks(batch_obs, pomdp):
    """
    finds all the auxiliary tasks in batch_obs
    """
    batch_size, n_foods, x_max, y_max = batch_obs.size()
    auxiliary_tasks_1 = torch.zeros(batch_size, n_foods, x_max, y_max)
    auxiliary_tasks_2 = torch.zeros(batch_size, n_foods, x_max, y_max)

    for ind in range(batch_size):
        task_1, task_2 = _augment_agent_color(batch_obs[ind,:,:,:], pomdp)
        auxiliary_tasks_1[ind, :, :, :] = task_1
        auxiliary_tasks_2[ind, :, :, :] = task_2

    return auxiliary_tasks_1, auxiliary_tasks_2

def _augment_agent_color(grid, pomdp):
    """
    changes the color of the agent and thus also its desire
    """
    grid = torch.unsqueeze(grid, 0)
    if pomdp:
        agent_x = 2
        agent_y = 2
    else:
        agent_cord = torch.where(grid[0, 0, :, :] == 1)
        agent_x = int(agent_cord[0][0])
        agent_y = int(agent_cord[1][0])

    grid_1 = grid.clone()
    grid_2 = grid.clone()

    grid_1[0, :, agent_x, agent_y] = torch.tensor([1, 0, 1, 0])
    grid_2[0, :, agent_x, agent_y] = torch.tensor([1, 0, 0, 1])

    return grid_1, grid_2
