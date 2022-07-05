import numpy as np
import random
from gym.spaces import Discrete 
import torch as T

from create_environemts import InitialStateDistribution
from env_params import EnvParams
from training import PPO

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
        self.size = env_params.size
        self.n_food_types = env_params.n_food_types
        self.n_actions = 5
        self.reward_range = (-np.inf, np.inf)
        self.pomdp = pomdp
        self.agent_desire = 1
        self.initial_state_distribution = InitialStateDistribution(env_params)

        self.test_set = self.initial_state_distribution.generate_states(env_params.n_test, 1)
        
        
        if pomdp:
            self.obs_size = (env_params.n_food_types + 1, obs_len*2 + 1, obs_len*2 + 1)
        else:
            self.obs_size = (env_params.n_food_types + 1, env_params.size[0], env_params.size[1])


    def reset(self):
        self.objectives = 0
        self.side_effects = 0
        self.grid = self.initial_state_distribution.generate_state()

        agent_cord = np.where(self.grid[0, :, :] == 1)
        agent_cord = (agent_cord[0][0], agent_cord[1][0])
        self.agent_cord = agent_cord

        obs = self.observe()
        return obs

    def observe(self):

        if self.pomdp:
            obs = self.get_observation()
            return obs

        else:
            return self.grid
            

    def get_observation(self):
        observation = np.zeros((self.n_food_types+1, self.obs_size[0], self.obs_size[1]))

        # for x and y relative to the agent
        for obs_x in range(self.obs_size[0]):
            for obs_y in range(self.obs_size[1]):
                grid_x = self.agent_cord[0] - int(self.obs_size[0]/2) + obs_x
                grid_y = self.agent_cord[1] - int(self.obs_size[1]/2) + obs_y

                if not self.inbounds(grid_x, grid_y):
                    observation[0, obs_x, obs_y] = 1
                else:
                    observation[:, obs_x, obs_y] = self.grid[:, grid_x, grid_y]

        # observation = T.flatten(T.tensor(observation, dtype=T.float))
        return observation



    def inbounds(self, x, y):
        return (-1 < x and x < self.size[0]) and (-1 < y and y < self.size[1])

    def step(self, action):
        direction = MOVEMENTS[action]
        agent_cord_new, agent_moved = self.move(self.agent_cord, direction)

        reward = -0.04
        if agent_moved:
            new_cell = self.grid[:, agent_cord_new[0], agent_cord_new[1]]
            print(new_cell)
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


env_params = EnvParams()
mdp = MDP(env_params)
ppo = PPO(mdp, 8, 16)
"""
env.reset()
env.step(1)
"""
# env.step(3)
