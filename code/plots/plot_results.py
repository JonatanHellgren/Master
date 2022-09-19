import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rcParams['text.usetex'] = True

width = 7
height = 6
dpi = 400 
xmin, xmax = (-0.3,2.1)
ymin, ymax = (0,3.1)

"""
legend = False
model_name = 'POMDP 8x8'
model = 'pomdp_8x8'

legend = True
model_name = 'POMDP 8x8 stochastic'
model = 'pomdp_8x8_stochastic'

legend = True
model_name = 'MDP 8x8'
model = 'static_8x8'
"""

legend = True
model_name = 'MDP 8x8 stochastic'
model = 'static_8x8_stochastic'
"""

legend = True
model_name = "POMDP 16x16"
model = "pomdp_16x16"

legend = True
model_name = "POMDP 16x16"
model = "pomdp_16x16"
"""

DIR = f'../models/{model}'


palette = sns.color_palette("colorblind")
col_ind = [0, 1, 2, 3, 4, 9, 8]
palette = [palette[i] for i in col_ind]

def summarize_data(DIR, top=30):

    # read data
    df = pd.read_csv(f'{DIR}/df.csv')

    # extract last training time steps
    df_top = df[df.time_step > top]

    data = defaultdict(list)

    lambda_range = load_lambda_range(DIR)
    for lmbda in lambda_range:
        df_top_lmbda = df_top[df_top['lambda'] == lmbda]

        mean_len = df_top_lmbda['avg_len'].mean() / 33
        mean_side_effect = df_top_lmbda['avg_side_effects'].mean()
        mean_obj = df_top_lmbda['avg_obj'].mean()

        data['lmbda'].append(lmbda)
        data['mean_len'].append(mean_len)
        data['mean_side_effect'].append(mean_side_effect)
        data['mean_obj'].append(mean_obj)

    return pd.DataFrame(data)

def load_lambda_range(DIR):
    with open(f'{DIR}/params.json') as json_file:
        data = json.load(json_file)['aux_params']
    return data['lambda_range']



def get_ax(height, width):
    cm = 1/2.54  # centimeters in inches
    fig = plt.figure(figsize=(width*cm,height*cm), dpi=dpi)
    ax = fig.add_subplot(1,1,1)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax),
           xticks=[0, 1, 2], xticklabels=['Length', 'Side effects', 'Objective'],
           yticks=[])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return ax

def plot_curve(start, finish, c='r', label=None):
    curve = (np.cos(np.linspace(-np.pi, 0))+1)/2

    start_x, start_y = start
    finish_x, finish_y = finish

    delta_y = finish_y - start_y
    modified_curve = curve * delta_y + start_y

    ax.plot(np.linspace(start_x, finish_x), modified_curve, c=c, label=label, alpha=0.8)

def plot_df_row(df_row):
    cord_1 = (0, df_row["mean_len"])
    cord_2 = (1, df_row["mean_side_effect"])
    cord_3 = (2, df_row["mean_obj"])

    color = palette[df_row.name]
    plot_curve(cord_1, cord_2, color, label=f"{df_row['lmbda']}")
    plot_curve(cord_2, cord_3, color)

def draw_axis(labels, xcord, ycords, ann_shift=0):
    plt.vlines(xcord, 0, 3, colors='black', alpha=0.5)
    for label, ycord in zip(labels, ycords):
        plt.annotate(label, (xcord-ann_shift, ycord))


def annotate_fig():
    labels_len = ['0 -', '20 -', '40 -', ' 60 -', '80 -', '100-']
    ycords = np.array([0, 0.6, 1.2, 1.8, 2.4, 3]) - 0.06
    draw_axis(labels_len, 0, ycords, 0.25)

    labels = ['- 0', '- 1', '- 2', '- 3']
    ycords = np.array([0, 1, 2, 3]) - 0.06
    draw_axis(labels, 1, ycords)
    draw_axis(labels, 2, ycords)

ax = get_ax(height, width)
annotate_fig()
df = summarize_data(DIR)
df.apply(plot_df_row, axis=1)
plt.suptitle(model_name)

if legend:
    plt.legend(title=r'$\lambda$', bbox_to_anchor=(1.05, 1.05), loc='upper left', prop={'size': 8})

plt.savefig(f'../../report/figures/{model}_results', bbox_inches='tight')

