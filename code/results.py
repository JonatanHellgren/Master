# TODO
# develop visualization so that it just includes plot_grid and print_grid
# add argparser to select a directory which plots and tables should be made from
# or plot which figure (add so that all can be plotted aswell)
# add simulation, or path thingy

import os

from visualization import save_table, plot_training_history, save_table_extra

for dir in os.listdir('./models'):
    save_table_extra(f'./models/{dir}')
    # plot_training_history(dir)

"""
Code for concatenating multiple dataset
"""
"""
import pandas as pd

DIR = 'models/pomdp_16x16_stochastic'
# df = pd.read_csv(f'{DIR}/df.csv')

df2 = pd.read_csv(f'{DIR}/df2.csv')
df3 = pd.read_csv(f'{DIR}/df3.csv')
df5 = pd.read_csv(f'{DIR}/df5.csv')

ind = ['critic_loss', 'avg_len', 'avg_obj', 'avg_side_effects',
       'dones', 'run', 'lambda', 'time_step']

# df2 = df2[~(df2['time_step'].isnull())]

df2 = df2[ind]
df3 = df3[ind]
df5 = df5[ind]

df3['run'] = df3['run'].apply(lambda x: x+2)
df5['run'] = df5['run'].apply(lambda x: x+5)

df = pd.concat([df2, df3, df5])
df.to_csv(f'{DIR}/df.csv')
"""
