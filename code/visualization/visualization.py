import itertools
import sys

import colorama
from colorama import Fore
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import numpy as np

from environment import MDP, EnvParams

palette = sns.color_palette("colorblind")
red = palette[1]
green = palette[2]
blue = palette[0]

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

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
            if cell[1] == 1:
                obj = plt.Rectangle((x_cord+0.15, y_cord+0.15), 0.7, 0.7, color=red)
            if cell[2] == 1:
                obj = plt.Rectangle((x_cord+0.15, y_cord+0.15), 0.7, 0.7, color=blue)
            if cell[3] == 1:
                obj = plt.Rectangle((x_cord+0.15, y_cord+0.15), 0.7, 0.7, color=green)

        elif cell[1] == 1:
            obj = plt.Circle(cord, 0.2, color=red)

        elif cell[2] == 1:
            obj = plt.Circle(cord, 0.2, color=blue)

        elif cell[3] == 1:
            obj = plt.Circle(cord, 0.2, color=green)

        axis.add_patch(obj)

def plot_grid(grid):
    _, max_x, max_y = grid.shape
    cm = 1/2.54  # centimeters in inches
    fig, axis = plt.subplots(figsize=(4*cm,4*cm), dpi=100)

    plot_grid_lines(axis, max_x, max_y)
    plot_objects(grid, axis, max_x, max_y)
    fig.show()

def print_grid(grid):

    if type(grid) is not torch.Tensor:
        grid = torch.tensor(grid)

    f, max_x, max_y = grid.size()

    _print_boarder(max_x)
    for y_cord in range(max_y):

        print(Fore.WHITE, '|', end='')
        for x_cord in range(max_x):
            cell = grid[:, y_cord, x_cord]

            # If agent cell
            if cell[0] == 1:
                if cell[1] == 1:
                    print(Fore.RED, '[=]', end='')
                    sys.stdout.flush()
                elif cell[2] == 1:
                    print(Fore.GREEN, '[=]', end='')
                elif cell[3] == 1:
                    print(Fore.BLUE, '[=]', end='')
                else:
                    print('[=]', end='')


            elif cell[1] == 1:
                print(Fore.RED, ' 0 ', end='')
            elif cell[2] == 1:
                print(Fore.GREEN, ' 0 ', end='')
            elif cell[3] == 1:
                print(Fore.BLUE, ' 0 ', end='')
            else:
                print('    ', end='')

            print(Fore.WHITE, '|', end='')
        
        print('\n')

        _print_boarder(max_x)

def _print_boarder(max_x):
    print(Fore.WHITE, '+', end='')
    for _ in range(max_x):
        print(Fore.WHITE, '--- +', end='')

    print('\n')


def run_environment():
    env_params = EnvParams(
            (4,4), # size
            9,      # n_foods
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

def plot_grid(grid, file_name):
    _, max_x, max_y = grid.shape
    cm = 1/2.54  # centimeters in inches
    fig, axis = plt.subplots(figsize=(4*cm,4.5*cm), dpi=300)
    plot_grid_lines(axis, max_x, max_y)
    plot_objects(grid, axis, max_x, max_y)
    plt.savefig(f'../report/figures/{file_name}')

# if __name__ == "__main__":
    # run_environment()
