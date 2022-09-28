import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

WIDTH = 6
HEIGHT = 4
DPI = 400 

palette = sns.color_palette("colorblind")
col_ind = [0, 1, 2, 4, 6, 7, 8]
palette = [palette[i] for i in col_ind]

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['text.usetex'] = True

def plot_training_history(dir):
    df = pd.read_csv(f'./models/{dir}/df.csv')
    df = df.reset_index()
    model = dir.split('/')[-1]

    cm = 1/2.54  # centimeters in inches
    fig, axis = plt.subplots(figsize=(WIDTH*cm,HEIGHT*cm), dpi=DPI)
    sns.lineplot(data=df, x='time_step', y='avg_side_effects', hue='lambda', palette=palette, legend=False)
    plt.xlabel('Timestep')
    plt.ylabel('Side effect')
    plt.suptitle(r"$\lambda$'s effect on side effects")
    plt.savefig(f'../report/figures/{model}_side_effects.png', bbox_inches='tight')

    fig, axis = plt.subplots(figsize=(WIDTH*cm,HEIGHT*cm), dpi=DPI)
    sns.lineplot(data=df, x='time_step', y='avg_obj', hue='lambda', palette=palette, legend=True)
    plt.xlabel('Timestep')
    plt.ylabel('Objective reward')
    plt.suptitle(r"$\lambda$'s effect on objective reward")
    plt.legend(title=r'$\lambda$', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f'../report/figures/{model}_objective_reward.png', bbox_inches='tight')

