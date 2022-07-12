import sys
from collections import defaultdict

from torch import torch, nn
from torch.distributions import Categorical
from torch.optim import Adam
import numpy as np

class PPO:
    def __init__(self, mdp, actor, critic, device, train_parameters):
        # Extract information from the environment
        self.mdp = mdp
        self.actor = actor
        self.critic = critic
        self.device = device
        self.train_parameters = train_parameters

        # Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.train_parameters.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.train_parameters.critic_lr)

        self.logging = defaultdict(list)

    def learn(self, n_epochs, total_timesteps, directory):
        """
        Trains the networks using the ppo clipped loss for n_epochs.
        Each epoch is not finished until the total timesteps are <= total_timesteps.
        After each epoch the test set is run through to evaluate the current networks.
        """
        lowest_len = 100
        for epoch in range(n_epochs):
            print(f'\nEpoch {epoch+1}:')
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
                value_estimate, _ = self.evaluate(batch_obs, batch_acts)

                # Compute advantage
                advantage = batch_rtgs - value_estimate.detach()

                # Normalize advantage to make the learning more stable
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
                # add small number to avoid zero division

                for _ in range(self.train_parameters.n_updates_per_iteration):
                    # Here we update the networks a few times with the current rollout
                    value_estimate, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                    actor_loss = _compute_actor_loss(curr_log_probs, batch_log_probs,\
                                                        advantage, self.train_parameters.clip)

                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.actor_optim.step()

                    critic_loss = nn.MSELoss()(value_estimate, batch_rtgs)

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()

            print('\n')
            avg_len, _, _ = self.run_test()

            if avg_len < lowest_len:
                print('Saving model...')
                lowest_len = avg_len
                torch.save(self.actor.state_dict(), f'./{directory}/best_actor')
                torch.save(self.critic.state_dict(), f'./{directory}/best_critic')

    def run_test(self):
        """
        Runs the test set defined in the mdp and returns the results
        """
        print("Running tests...")
        lengths = []
        objectives = []
        side_effects = []
        dones = 0
        with torch.no_grad():
            for test in self.mdp.test_set:

                obs = self.mdp.set_initial_state(np.copy(test))
                done = False
                time_step = 0
                for _ in range(100):
                    time_step += 1

                    action, _ = self.get_action(obs, greedy=True)
                    obs, _, done, _ = self.mdp.step(int(action))

                    if done:
                        dones += 1
                        break

                lengths.append(time_step)
                objectives.append(self.mdp.objectives)
                side_effects.append(self.mdp.side_effects)

        avg_len = round(np.mean(lengths), 2)
        avg_obj = round(np.mean(objectives), 2)
        avg_side_effects = round(np.mean(side_effects), 2)

        self.logging['avg_len'].append(avg_len)
        self.logging['avg_obj'].append(avg_obj)
        self.logging['avg_side_effects'].append(avg_side_effects)
        self.logging['dones'].append(dones)
        print(f"avg_len: {avg_len}\n avg_obj: {avg_obj}\n avg_side_effects: {avg_side_effects}\n\
                done: {dones}/100")

        return avg_len, avg_obj, avg_side_effects

    def rollout(self):
        """
        Move out rollout, make usable for manager aswell
        """
        # Batch data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        time_step = 0

        while time_step < self.train_parameters.timesteps_per_batch:
            # rewards from episode
            ep_rews = []
            obs = self.mdp.reset()
            done = False
            for ep_time_step in range(self.train_parameters.max_timesteps_per_episode):
                time_step += 1

                # Collect observation
                batch_obs.append(np.copy(obs))

                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.mdp.step(int(action))

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
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.long)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.train_parameters.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, obs, greedy=False):
        """
        Getting an action from the actor given an observation.
        Either using the stochastic policy or the greedy one.
        """
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
        value_estimate = self.critic(batch_obs).squeeze()

        all_probs = self.actor(batch_obs)
        log_probs = torch.log(all_probs[range(len(batch_acts)), batch_acts])

        return value_estimate, log_probs

def _compute_actor_loss(curr_log_probs, batch_log_probs, advantage, clip):
    # Since, exp(log(a) - log(b)) = (a / b), we can perform this computation
    ratios = torch.exp(curr_log_probs - batch_log_probs)
    surr1 = ratios * advantage
    surr2 = torch.clamp(ratios, 1 - clip, 1 + clip) * advantage

    actor_loss = (-torch.min(surr1, surr2)).mean()

    return actor_loss
