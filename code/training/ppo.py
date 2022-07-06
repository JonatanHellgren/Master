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

        # Find what device is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initilize actor 
        self.actor = FeedForwardNN(self.obs_dim, n_conv, hidden_dim,
                self.act_dim, self.device, softmax=True).to(self.device)
        # and critic
        self.critic = FeedForwardNN(self.obs_dim, n_conv, hidden_dim,
                1, self.device).to(self.device)

        # Where are the hyper parameters? Here they are!
        self._init_hyperparameters()

        # Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)


    def learn(self, n_epochs, total_timesteps):
        """
        Trains the networks using the ppo clipped loss for n_epochs.
        Each epoch is not finished until the total timesteps are <= total_timesteps.
        After each epoch the test set is run through to evaluate the current networks.
        """

        for it in range(n_epochs):
            print(f'\nEpoch {it+1}:')
            t_so_far = 0 

            while t_so_far < total_timesteps:
                # yank progress bar
                print('.', end='')
                sys.stdout.flush()

                # Performing a rollout and moving all the information to the gpu, except batch_lens
                batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
                batch_obs = batch_obs.to(self.device)
                batch_acts = batch_acts.to(self.device)
                batch_log_probs = batch_log_probs.to(self.device)
                batch_rtgs = batch_rtgs.to(self.device)

                # We only need batch_lens to count the total amount of steps
                t_so_far += np.sum(batch_lens)

                # Critics evalutations
                V, _ = self.evaluate(batch_obs, batch_acts)

                # Compute advantage
                A_k = batch_rtgs - V.detach()

                # Normalize advantage to make the learning more stable
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) # add small number to avoid zero division

                for _ in range(self.n_updates_per_iteration):
                    # Here we update the networks a few times with the current rollout
                    V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                    # Since, exp(log(a) - log(b)) = (a / b), we can perform this computation
                    ratios = torch.exp(curr_log_probs - batch_log_probs)
                    surr1 = ratios * A_k
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                    actor_loss = (-torch.min(surr1, surr2)).mean()

                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    # print(f"actor loss: {actor_loss}")
                    self.actor_optim.step()

                    critic_loss = nn.MSELoss()(V, batch_rtgs)

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    # print(f"critic loss: {critic_loss}")
                    self.critic_optim.step()

            print('\n')
            self.run_test()


    def run_test(self):
        print("Running tests...") 
        lengths = []
        objectives = []
        side_effects = []
        with torch.no_grad():
            for test in self.env.test_set:

                obs = self.env.set_initial_state(np.copy(test))
                done = False
                t = 0
                for _ in range(self.max_timesteps_per_episode):
                    t += 1

                    action, _ = self.get_action(obs, greedy=True)
                    obs, _, done, _ = self.env.step(int(action))

                    if done:
                        break

                lengths.append(t)
                objectives.append(self.env.objectives)
                side_effects.append(self.env.side_effects)

        print(f"avg_len: {round(np.mean(lengths), 2)}\n avg_obj: {round(np.mean(objectives), 2)}\n avg_side_effects: {round(np.mean(side_effects), 2)}")


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
                batch_obs.append(np.copy(obs))

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
        self.timesteps_per_batch = 500
        self.max_timesteps_per_episode = 500 
        self.gamma = 0.95
        self.n_updates_per_iteration = 3
        self.clip = 0.2
        self.actor_lr = 1e-4
        self.critic_lr = 7e-4

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



            


