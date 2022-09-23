import numpy as np
from numpy import random

class InitialGridDistribution:
    """
    This class represents the initial state distribution.
    It is able to generate a single random state, and a set of states.
    """

    def __init__(self, env_params, agent_desire):
        self.env_params = env_params
        self.agent_desire = agent_desire

        max_x, max_y = env_params.size
        self.all_cords = [(x, y) for x in range(max_x) for y in range(max_y)]

    def generate_grid(self):
        """
        Generates a single state by drawing from the initial state distribution
        """
        grid = np.zeros([self.env_params.n_food_types+1, self.env_params.size[0],
                          self.env_params.size[1]])
        cord_inds = random.choice(len(self.all_cords), self.env_params.n_foods+1, replace=False)
        cords = [self.all_cords[ind] for ind in cord_inds]

        # Placing the agent at the first cord with its desire (one)
        agent_cord = cords.pop(0)
        grid[0, agent_cord[0], agent_cord[1]] = self.agent_desire
        grid[1, agent_cord[0], agent_cord[1]] = self.agent_desire

        for ind, cord in enumerate(cords):
            # food types cycles
            food_type = ind % 3 + 1
            # Place food
            grid[food_type, cord[0], cord[1]] = 1

        return grid

    def generate_grids(self, n_grids, seed):
        """
        Generates a set of states drawn from the initial state distribution using a seed
        """
        random.seed(seed=seed) # So many seeds!

        grids = []
        for _ in range(n_grids):
            grid = self.generate_grid()
            grids.append(grid)

        return grids
