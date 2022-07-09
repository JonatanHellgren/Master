import numpy as np
from numpy import random

class InitialStateDistribution:
    """
    This class represents the initial state distribution.
    It is able to generate a single random state, and a set of states.
    """

    def __init__(self, env_params, agent_desire):
        self.env_params = env_params
        self.agent_desire = agent_desire

        max_x, max_y = env_params.size
        self.all_cords = [(x, y) for x in range(max_x) for y in range(max_y)]

    def generate_state(self):
        """
        Generates a single state by drawing from the initial state distribution
        """
        state = np.zeros([self.env_params.n_food_types+1, self.env_params.size[0],
                          self.env_params.size[1]])
        cord_inds = random.choice(len(self.all_cords), self.env_params.n_foods+1, replace=False)
        cords = [self.all_cords[ind] for ind in cord_inds]

        # Placing the agent at the first cord with its desire (one)
        agent_cord = cords.pop(0)
        state[0, agent_cord[0], agent_cord[1]] = self.agent_desire
        state[1, agent_cord[0], agent_cord[1]] = self.agent_desire

        for ind, cord in enumerate(cords):
            # First three will be of type one
            if ind < 3:
                food_type = self.agent_desire
            # Then we sample randomly from a food greater then one
            else:
                food_type = random.choice(range(1, self.env_params.n_food_types)) + 1
            # Place food
            state[food_type, cord[0], cord[1]] = 1

        return state

    def generate_states(self, n_states, seed):
        """
        Generates a set of states drawn from the initial state distribution using a seed
        """
        random.seed(seed=seed) # So many seeds!

        states = []
        for _ in range(n_states):
            state = self.generate_state()
            states.append(state)

        return states
