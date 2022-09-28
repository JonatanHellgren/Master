import numpy as np
import matplotlib.pyplot as plt

from visualization import plot_grid
from environment import MDP, EnvParams

def fig_stochastic():
    grid = np.zeros([4, 5, 5])

    grid[0, 1, 1] = 1
    grid[1, 1, 1] = 1
    grid[1, 4, 2] = 1

    plot_grid(grid, 'stoch.png')

    # move agent
    grid[0, 1, 1] = 0
    grid[1, 1, 1] = 0
    grid[0, 2, 1] = 1
    grid[1, 2, 1] = 1

    plot_grid(grid, 'noop.png')

    # move food down
    grid[1, 4, 2] = 0
    grid[1, 4, 3] = 1

    plot_grid(grid, 'down.png')

    # move food left
    grid[1, 4, 3] = 0
    grid[1, 3, 2] = 1

    plot_grid(grid, 'left.png')

    # move food up
    grid[1, 3, 2] = 0
    grid[1, 4, 1] = 1

    plot_grid(grid, 'up.png')

fig_stochastic()

def fig_3_1():
    grid = np.zeros([4, 5, 5])

    grid[0, 1, 1] = 1
    grid[1, 1, 1] = 1
    grid[1, 2, 2] = 1

    plot_grid(grid, 's1.png')

    grid[0, 1, 1] = 0
    grid[1, 1, 1] = 0
    grid[0, 2, 1] = 1
    grid[1, 2, 1] = 1

    plot_grid(grid, 's2.png')

# fig_3_1()

def fig_3_2():
    env_params = EnvParams(
            (8,8), # size
            15,      # n_foods
            3,       # n_food_types
            False,
            False,
            100)     # n_test

    mdp = MDP(env_params)
    grid = mdp.reset()
    grid[1, 0, 6] = 0
    grid[1, 3, 3] = 1

    plot_grid(grid, '5x5.png')

    grid = mdp.step(1)[0]

    plot_grid(grid, '5x5_2.png')


# fig_3_2()

def aux_example():
    env_params = EnvParams(
            (8,8), # size
            15,      # n_foods
            3,       # n_food_types
            False,
            False,
            100)     # n_test

    mdp = MDP(env_params)
    grid = mdp.reset()

    grid[1, 0, 6] = 0
    grid[1, 3, 3] = 1

    grid[1,3,2] = 0
    grid[2,3,2] = 1
    plot_grid(grid, 'aux_1.png')

    grid[2,3,2] = 0
    grid[3,3,2] = 1
    plot_grid(grid, 'aux_2.png')

# aux_example()

"""
cm = 1/2.54  # centimeters in inches
fig, axis = plt.subplots(figsize=(4*cm,4.5*cm), dpi=300)
plot_grid_lines(axis, max_x, max_y)
plot_objects(grid, axis, max_x, max_y)
# fig.show()
plt.savefig('../report/figures/s1.png')
"""



