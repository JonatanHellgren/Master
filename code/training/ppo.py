import sys

import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam
import numpy as np

from training.networks import FeedForwardNN

class PPO:
    def __init__(self, env, n_conv, hidden_dim):
        # Extract information from the environment
        self.env = env
        self.obs_dim = env.obs_size
        self.act_dim = env.n_actions

        # Initilize actor and critic
        self.actor = FeedForwardNN(self.obs_dim, n_conv, hidden_dim, self.act_dim, softmax=True)
        self.critic = FeedForwardNN(self.obs_dim, n_conv, hidden_dim, 1)

        self._init_hyperparameters()

        # Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)


    def learn(self, n_epochs, total_timesteps):

        for it in range(n_epochs):
            print(f'\nEpoch {it}:')
            t_so_far = 0 

            while t_so_far < total_timesteps:
                print('.', end='')
                sys.stdout.flush()

                batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
                print(np.mean(batch_lens))

                t_so_far += np.sum(batch_lens)

                # Critics evalutations
                V, _ = self.evaluate(batch_obs, batch_acts)

                # Compute advantage
                A_k = batch_rtgs - V.detach()

                # Normalize advantage
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) # add small number to avoid zero division

                for _ in range(self.n_updates_per_iteration):
                    V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                    ratios = torch.exp(curr_log_probs - batch_log_probs)

                    surr1 = ratios * A_k
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                    actor_loss = (-torch.min(surr1, surr2)).mean()

                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.actor_optim.step()

                    critic_loss = nn.MSELoss()(V, batch_rtgs)

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()




    def rollout(self):
        # Batch data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        t = 0

        while t < self.timesteps_per_batch:
            # rewards from episode
            ep_rews = []
            obs = self.env.reset()
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                # Collect observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(int(action))

                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # +1, since t is initalized as 0
            batch_rews.append(ep_rews)

        # Convert batch data to tensors
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.long)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens


    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.1
        self.actor_lr = 1e-5
        self.critic_lr = 1e-4

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            
            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs


    def get_action(self, obs, greedy=False):

        probs = self.actor(obs)

        if greedy:
            # Select action with highest probability
            action = torch.argmax(probs, dim=1)
        else: 
            # Sample action
            distr = Categorical(probs)
            action = distr.sample()

        log_prob = torch.log(probs[0, action])

        return action, log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        
        all_probs = self.actor(batch_obs)
        log_probs = torch.log(all_probs[range(len(batch_acts)), batch_acts])

        return V, log_probs



            


