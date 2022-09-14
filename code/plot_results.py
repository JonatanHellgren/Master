import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# plt.rcParams['text.usetex'] = True

palette = sns.color_palette("colorblind")
col_ind = [0, 1, 2, 4, 3, 5, 8]
palette = [palette[i] for i in col_ind]

fake_data = {
        "lambda": [0, 1, 2, 3],
        "side_effects": [1.5, 0.7, 0.3, 0],
        "objective_reward": [3, 3, 2.5, 0]
        }

fake_df = pd.DataFrame(fake_data)

curve = (np.cos(np.linspace(-np.pi, 0))+1)/2
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

def plot_curve(start, finish, c='r'):

    start_x, start_y = start
    finish_x, finish_y = finish

    delta_y = finish_y - start_y
    modified_curve = curve * delta_y + start_y

    ax.plot(np.linspace(start_x, finish_x), modified_curve, c=c)

def plot_df_row(df_row):
    cord_1 = (0, df_row["lambda"])
    cord_2 = (1, df_row["side_effects"])
    cord_3 = (2, df_row["objective_reward"])

    color = palette[df_row.name]
    plot_curve(cord_1, cord_2, color)
    plot_curve(cord_2, cord_3, color)

# plot_curve((0,0), (1,2), 'r')

# plot_curve((1,2), (2,0), 'r')
fake_df.apply(plot_df_row, axis=1)

xmin, xmax = (-0.1,2.1)
ymin, ymax = (-0.1,3.1)
ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax),
       xticks=[0, 1, 2], xticklabels=['lambda', 'Side effects', 'Objective reward'])
plt.show()

