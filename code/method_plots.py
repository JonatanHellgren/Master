import numpy as np
import matplotlib.pyplot as plt

from visualization import plot_grid
from environment import MDP, EnvParams


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

fig_3_1()

def fig_3_2():
    env_params = EnvParams(
            (5,5), # size
            9,      # n_foods
            3,       # n_food_types
            100)     # n_test

    mdp = MDP(env_params)
    grid = mdp.reset()

    plot_grid(grid, '5x5.png')

    grid = mdp.step(1)[0]

    plot_grid(grid, '5x5_2.png')


fig_3_2()

"""
cm = 1/2.54  # centimeters in inches
fig, axis = plt.subplots(figsize=(4*cm,4.5*cm), dpi=300)
plot_grid_lines(axis, max_x, max_y)
plot_objects(grid, axis, max_x, max_y)
# fig.show()
plt.savefig('../report/figures/s1.png')
"""



