import numpy as np
from numpy import random

class InitialStateDistribution:
    def __init__(self, env_params):
        self.env_params = env_params

        max_x, max_y = env_params.size
        self.all_cords = [(x, y) for x in range(max_x) for y in range(max_y)]

    def generate_states(self, n_states, seed):
        random.seed(seed=seed) # So many seeds!

        states = []
        for _ in range(n_states):
            state = self.generate_state()
            states.append(state)

        return states

    def generate_state(self):
        state = np.zeros([self.env_params.n_food_types+1, self.env_params.size[0], self.env_params.size[1]])
        cord_inds = random.choice(len(self.all_cords), self.env_params.n_foods+1, replace=False)
        cords = [self.all_cords[ind] for ind in cord_inds]

        # agent
        agent_cord = cords.pop(0)
        state[0, agent_cord[0], agent_cord[1]] = 1
        state[1, agent_cord[0], agent_cord[1]] = 1

        for ind, cord in enumerate(cords):
            if ind < 3:
                food_type = 1
            else:
                food_type = random.choice(range(1, self.env_params.n_food_types)) + 1

            state[food_type, cord[0], cord[1]] = 1

        return state
"""
environment_factory = InitialStateDistribution(env_params)
S = environment_factory.generate_states(10, 10)
for s in S:
    print(np.sum(s)) # == 11 if env_params.n_foods = 9
    print(np.sum(s[0,:,:])) # == 1, only one agent
    print(np.sum(s[1,:,:])) # == 4, three foods and one agent

"""
