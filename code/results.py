# TODO
# develop visualization so that it just includes plot_grid and print_grid
# add argparser to select a directory which plots and tables should be made from
# or plot which figure (add so that all can be plotted aswell)
# add simulation, or path thingy

import os

from visualization import save_table, plot_training_history

# for dir in os.listdir('./models'):
    # save_table(f'./models/{dir}')
    # plot_training_history(dir)

import pandas as pd

df = pd.read_csv(f'/df.csv')

