import itertools

import matplotlib.pyplot as plt
import numpy as np

from environment import MDP, EnvParams

def plot_grid_lines(axis, max_x, max_y):
    """
    Creates a empty grid on the given axis of the specified size (max_x, max_y).
    """
    axis.set_ylim(0, max_x)
    axis.set_xlim(0, max_y)

    x_labels = [int(np.floor(n)) if np.floor(n) != n else '' for n in np.linspace(0, max_x, max_x*2+1)]
    y_labels = [int(np.floor(n)) if np.floor(n) != n else '' for n in np.linspace(0, max_y, max_y*2+1)]

    axis.set_xticks(ticks=np.linspace(0,max_x,max_x*2+1), labels=x_labels)
    axis.set_yticks(ticks=np.linspace(0,max_y,max_y*2+1), labels=y_labels)

    for ind in range(1, max_x):
        axis.axvline(ind, color='black', alpha=0.2)

    for ind in range(1, max_y):
        axis.axhline(ind, color='black', alpha=0.2)

    axis.set_aspect('equal', adjustable='box')

def plot_objects(grid, axis, max_x, max_y):
    """
    Loops through the grid and plots all objects in it
    """
    for x_cord, y_cord in itertools.product(range(max_x), range(max_y)):
        cell = grid[:, x_cord, y_cord]
        plot_object(cell, x_cord, y_cord, axis)

def plot_object(cell, x_cord, y_cord, axis):
    """
    Plots a single object in a cell with its correct shape and color
    """
    if sum(cell) > 0:
        cord = (x_cord+0.5, y_cord+0.5)

        # Agent
        if cell[0] == 1:
            obj = plt.Rectangle((x_cord+0.1, y_cord+0.1), 0.8, 0.8, color='r')

        elif cell[1] == 1:
            obj = plt.Circle(cord, 0.3, color='r')

        elif cell[2] == 1:
            obj = plt.Circle(cord, 0.3, color='b')

        elif cell[3] == 1:
            obj = plt.Circle(cord, 0.3, color='g')

        axis.add_patch(obj)

def plot_grid(grid):
    _, max_x, max_y = grid.shape
    fig, axis = plt.subplots(figsize=(1,1))

    plot_grid_lines(axis, max_x, max_y)
    plot_objects(grid, axis, max_x, max_y)
    fig.show()

def run_environment():
    env_params = EnvParams(
            (10,10), # size
            15,      # n_foods
            3,       # n_food_types
            100)     # n_test
    mdp = MDP(env_params)
    grid = mdp.reset()
    _, max_x, max_y = grid.shape
    fig, axis = plt.subplots(figsize=(1,1))

    done = False
    while not done:

        # add side effects and objectives to mdp.info and teturn with .step
        plot_grid_lines(axis, max_x, max_y)
        plot_objects(grid, axis, max_x, max_y)
        fig.show()

        action = input('Action: ')
        action = int(action)
        grid, reward, done, info = mdp.step(action)

        axis.clear()
        if done:
            plot_grid(axis, max_x, max_y)
            plot_objects(grid, axis, max_x, max_y)
            fig.show()

# if __name__ == "__main__":
    # run_environment()
