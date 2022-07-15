import sys
from collections import defaultdict

from torch import save, no_grad, squeeze, nn
import numpy as np
from numpy import random

from .training_utils import rollout, rollout_test_set

class ManagerTrainer:

    def __init__(self, mdp, agent, device, train_parameters):
        self.mdp = mdp
        self.agent = agent
        self.device = device
        self.train_parameters = train_parameters

        self.logging = defaultdict(list)

        self.x_test, self.y_test = self.get_test_set()
        self.y_test = self.y_test.to(device)

        # Make sure that manager is incuded in agent
        assert agent.manager is not None

    def train(self, n_epochs, total_timesteps, directory):

        lowest_loss = 10
        for epoch in range(n_epochs):
            print(f'\nEpoch {epoch+1}:')
            t_so_far = 0

            while t_so_far < total_timesteps:
                # yank progress bar
                print('.', end='')
                sys.stdout.flush()

                # Performing a rollout and moving all the information to the gpu, except batch_lens
                batch_obs, _, _, batch_rtgs, batch_lens = \
                        rollout(self.agent, self.train_parameters, self.mdp)

                # Augment the batch
                _augment_batch(batch_obs)

                # Move to gpu
                batch_obs = batch_obs.to(self.device)
                batch_rtgs = batch_rtgs.to(self.device)

                # We only need batch_lens to count the total amount of steps
                t_so_far += np.sum(batch_lens)

                self.agent.train_manager(batch_obs, batch_rtgs)

            manager_loss = self.run_test()
            self.logging["loss"].append(manager_loss)
            if manager_loss < lowest_loss:
                lowest_loss = manager_loss
                print('Saving model...')
                save(self.agent.manager.state_dict(), f'./{directory}/best_manager')

    def get_test_set(self):
        batch_obs, batch_rtgs, _, _, _, _ = \
                rollout_test_set(self.agent, self.train_parameters, self.mdp)

        return batch_obs, batch_rtgs

    def run_test(self):
        with no_grad():
            manager_rtgs = squeeze(self.agent.manager(self.x_test))
            manager_loss = nn.MSELoss()(manager_rtgs, self.y_test)
            print(manager_loss)
            return manager_loss


def _augment_batch(batch_obs):

    for ind in range(batch_obs.size()[0]):
        aug_dim = random.choice([1, 2, 3], 3, replace=False)
        aug_dim = np.insert(aug_dim, 0, 0)
        batch_obs[ind, :, :, :] = batch_obs[ind, aug_dim, :, :]
