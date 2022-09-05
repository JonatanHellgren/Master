import sys
from collections import defaultdict

from torch import save, no_grad
import numpy as np

from .training_utils import rollout, rollout_test_set

class PPO:
    """
    Training algorithm
    """
    def __init__(self, mdp, agent, device, train_parameters, use_aux=False, save_model=True):
        # Extract information from the environment
        self.mdp = mdp
        self.agent = agent
        self.device = device
        self.train_parameters = train_parameters
        self.use_aux = use_aux
        self.save_model = save_model

        self.logging = defaultdict(list)

    def train(self, n_epochs, total_timesteps, directory):
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
                batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = \
                        rollout(self.agent, self.train_parameters, self.mdp, self.use_aux)

                # Move to gpu
                batch_obs = batch_obs.to(self.device)
                batch_acts = batch_acts.to(self.device)
                batch_log_probs = batch_log_probs.to(self.device)
                batch_rtgs = batch_rtgs.to(self.device)

                # We only need batch_lens to count the total amount of steps
                t_so_far += np.sum(batch_lens)

                self.agent.train(batch_obs, batch_acts, batch_log_probs, batch_rtgs)

            print('\n')
            avg_len = self.run_test()

            if avg_len < lowest_len and self.save_model:
                lowest_len = avg_len
                self.save_models(directory)

    def save_models(self, directory):
        """
        Saves the current models
        """
        print('Saving model...')
        save(self.agent.actor.state_dict(), f'./{directory}/best_actor')
        save(self.agent.critic.state_dict(), f'./{directory}/best_critic')

    def run_test(self):

        _, _, avg_len, avg_obj, avg_side_effects, dones, _ = \
                rollout_test_set(self.agent, self.train_parameters, self.mdp)
        self.logging['avg_len'].append(avg_len)
        self.logging['avg_obj'].append(avg_obj)
        self.logging['avg_side_effects'].append(avg_side_effects)
        self.logging['dones'].append(dones)
        print(f"avg_len: {avg_len}\n avg_obj: {avg_obj}\n avg_side_effects: {avg_side_effects}\n\
                done: {dones}/100")

        return avg_len
