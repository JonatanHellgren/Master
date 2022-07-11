import numpy as np

from environment.initial_grid_distribution import InitialGridDistribution

MOVEMENTS = {
    0: ( 0, 0), # Noop
    1: (-1, 0), # Up
    2: ( 0, 1), # Right
    3: ( 1, 0), # Down
    4: ( 0,-1)  # Left
        }

class MDP:
    def __init__(self, env_params, obs_len=2, pomdp=False):

        self.env_params = env_params
        self.n_actions = 5
        self.pomdp = pomdp
        self.agent_desire = 1
        self.initial_grid_distribution = InitialGridDistribution(env_params, self.agent_desire)

        self.test_set = self.initial_grid_distribution.generate_grids(env_params.n_test, 1)

        if pomdp:
            self.obs_size = (env_params.n_food_types + 1, obs_len*2 + 1, obs_len*2 + 1)
        else:
            self.obs_size = (env_params.n_food_types + 1, env_params.size[0], env_params.size[1])


    def reset(self):
        self.objectives = 0
        self.side_effects = 0
        self.grid = self.initial_grid_distribution.generate_grid()

        agent_cord = np.where(self.grid[0, :, :] == 1)
        agent_cord = (agent_cord[0][0], agent_cord[1][0])
        self.agent_cord = agent_cord

        obs = self.observe()
        return obs

    def set_initial_state(self, grid):
        self.objectives = 0
        self.side_effects = 0
        self.grid = grid

        agent_cord = np.where(self.grid[0, :, :] == 1)
        agent_cord = (agent_cord[0][0], agent_cord[1][0])
        self.agent_cord = agent_cord

        obs = self.observe()
        return obs

    def observe(self):

        if self.pomdp:
            obs = self.get_observation()
            return obs
        return self.grid

    def get_observation(self):
        observation = np.zeros(self.obs_size)

        # for x and y relative to the agent
        for obs_x in range(self.obs_size[1]):
            for obs_y in range(self.obs_size[2]):
                grid_x = self.agent_cord[0] - int(self.obs_size[0]/2) + obs_x
                grid_y = self.agent_cord[1] - int(self.obs_size[1]/2) + obs_y

                if not self.inbounds(grid_x, grid_y):
                    observation[0, obs_x, obs_y] = 1
                else:
                    observation[:, obs_x, obs_y] = self.grid[:, grid_x, grid_y]

        # observation = T.flatten(T.tensor(observation, dtype=T.float))
        return observation



    def inbounds(self, x_cord, y_cord):
        return (-1 < x_cord and x_cord < self.env_params.size[0]) and (-1 < y_cord and y_cord < self.env_params.size[1])

    def step(self, action):
        direction = MOVEMENTS[action]
        agent_cord_new, agent_moved = self.move(self.agent_cord, direction)

        reward = -0.04
        if agent_moved and action != 0:
            new_cell = self.grid[:, agent_cord_new[0], agent_cord_new[1]]
            agent_cell = np.copy(self.grid[:, self.agent_cord[0], self.agent_cord[1]])
            if any(new_cell == 1):
                food_type = np.where(new_cell==1)[0][0]
                if food_type == self.agent_desire:
                    self.objectives += 1
                    reward = 1
                else:
                    self.side_effects += 1
            self.grid[:, self.agent_cord[0], self.agent_cord[1]] = np.zeros(self.env_params.n_food_types+1)
            self.grid[:, agent_cord_new[0], agent_cord_new[1]] = agent_cell
            self.agent_cord = agent_cord_new

        obs = self.observe()
        done = self.objectives > 2
        info = None

        return obs, reward, done, info 


    def move(self, cord, direction):
        cord_new = (cord[0] + direction[0], cord[1] + direction[1])
        if self.inbounds(cord_new[0], cord_new[1]):
            return cord_new, True
        else:
            return cord, False


