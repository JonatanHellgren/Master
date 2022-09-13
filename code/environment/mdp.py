import numpy as np

from environment.initial_grid_distribution import InitialGridDistribution

# Dict for the cardinal directions
MOVEMENTS = {
    0: ( 0, 0), # Noop
    1: ( 0, 1), # Up
    2: ( 1, 0), # Right
    3: ( 0,-1), # Down
    4: (-1, 0)  # Left
            }

class MDP:
    """
    Class for the environment
    Named MDP, but extending it to a POMDP is possible by setting pomdp=True. If this is
    the case, then it is also possible to select observation length, this length is how far 
    the agent will be able to see in all directions including diagonal. Thus it defines a 
    square with the side equal to the length of obs_len*2+1 cells with the agent in the center.
    """
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
        """
        Resets the MDP and returns a new grid from the initial grid distribution
        """
        self.objectives = 0
        self.side_effects = 0
        self.grid = self.initial_grid_distribution.generate_grid()

        agent_cord = np.where(self.grid[0, :, :] == 1)
        agent_cord = (0, agent_cord[0][0], agent_cord[1][0])
        self.agent_cord = agent_cord

        obs = self.observe()
        return obs

    def set_initial_state(self, grid):
        """
        Function for manualy setting reseting the MDP, this lets the user choose which
        grid will be used in the inital state
        """
        self.objectives = 0
        self.side_effects = 0
        self.grid = grid

        agent_cord = np.where(self.grid[0, :, :] == 1)
        agent_cord = (0, agent_cord[0][0], agent_cord[1][0])
        self.agent_cord = agent_cord

        obs = self.observe()
        return obs

    def observe(self):
        """
        This functions returns the agents observation of the grid
        """
        if self.pomdp:
            obs = self.get_observation()
            return obs
        return self.grid

    def get_observation(self):
        """
        If POMDP, then we need to collect all the cells visible for the agent. 
        If the agent is close to the edge of the grid, then all cells 'visible' outside
        the edge will be equal to 1 in the 0th dimention.
        """
        observation = np.zeros(self.obs_size)

        # for x and y relative to the agent
        for obs_x in range(self.obs_size[1]):
            for obs_y in range(self.obs_size[2]):
                grid_x = self.agent_cord[1] - int(self.obs_size[0]/2) + obs_x
                grid_y = self.agent_cord[2] - int(self.obs_size[1]/2) + obs_y

                if not self.inbounds((0, grid_x, grid_y)):
                    observation[0, obs_x, obs_y] = 1
                else:
                    observation[:, obs_x, obs_y] = self.grid[:, grid_x, grid_y]

        # observation = T.flatten(T.tensor(observation, dtype=T.float))
        return observation

    def inbounds(self, cord):
        """ Function to check if cord exsists in the grid """
        _, x_cord, y_cord = cord
        return (-1 < x_cord and x_cord < self.env_params.size[0]) and \
                (-1 < y_cord and y_cord < self.env_params.size[1])

    def step(self, action):
        """
        This function executes a timestep in the environment. It moves the agent and 
        checks if any food has been consumed.
        """
        agent_cord_new, agent_moved = self.get_new_cord(self.agent_cord, action)

        if agent_moved and action != 0:
            reward = self.move_agent(agent_cord_new)
        else:
            reward = -0.04
        # breakpoint()

        if self.env_params.stochastic:
            self.move_foods()

        obs = self.observe()
        done = self.objectives > 2
        info = None

        return obs, reward, done, info 

    def move_agent(self, agent_cord_new):

        new_cell = self.grid[:, agent_cord_new[1], agent_cord_new[2]]
        agent_cell = np.copy(self.grid[:, self.agent_cord[1], self.agent_cord[2]])
        if any(new_cell == 1):
            food_type = np.where(new_cell==1)[0][0]
            if food_type == self.agent_desire:
                self.objectives += 1
                reward = 1
            else:
                self.side_effects += 1
                reward = -0.04
        else:
            reward = -0.04

        self.move(self.agent_cord, agent_cord_new)
        # self.grid[:, self.agent_cord[0], self.agent_cord[1]] = np.zeros(self.env_params.n_food_types+1)
        # self.grid[:, agent_cord_new[0], agent_cord_new[1]] = agent_cell
        self.agent_cord = agent_cord_new
        return reward
    
    def move_foods(self):
        food_cords = self.get_food_cords()

        for food_cord in food_cords:
            self.move_food(food_cord)

    def move_food(self, food_cord):
        # breakpoint()
        action = np.random.choice(4)
        new_cord, new = self.get_new_cord(food_cord, action)
        if new and all(self.grid[:, new_cord[1], new_cord[2]] != 1):
            self.move(food_cord, new_cord)


    def get_food_cords(self):
        ones = np.where(self.grid[1:]==1) # all locations where the grid equals 1
        n_foods = len(ones[0]) - 1 # since agent is included
        food_cords = [(ones[0][it]+1, ones[1][it], ones[2][it]) for it in range(n_foods)]

        return food_cords

    def get_new_cord(self, cord, action):
        """
        Inputs a cord and a direction
        Returns the new cord and boolean representing if the cord changed
        """
        direction = MOVEMENTS[action]
        if self.inbounds(cord) and action != 0:
            cord_new = (cord[0], cord[1] + direction[0], cord[2] + direction[1])
            return cord_new, True
        else:
            return cord, False

    def move(self, cord, new_cord):
        # breakpoint()
        f = np.shape(self.grid)[0]
        old_cell = np.copy(self.grid[:, cord[1], cord[2]])
        self.grid[:, cord[1], cord[2]] = np.zeros(f)
        self.grid[:, new_cord[1], new_cord[2]] = old_cell

