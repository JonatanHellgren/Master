import pandas as pd
import matplotlib.pyplot as plt


WIDTH = 6
HEIGHT = 4
DPI = 100 

def plot_results(dir):
    df = pd.read_csv('models/static_8x8/df.csv')
    df = df.reset_index()

    cm = 1/2.54  # centimeters in inches
    fig, axis = plt.subplots(figsize=(WIDTH*cm,HEIGHT*cm), dpi=DPI)
    sns.lineplot(data=df, x='time_step', y='avg_side_effects', hue='lambda', palette=palette, legend=False)
    plt.xlabel('Timestep')
    plt.ylabel('Side effect')
    plt.suptitle(r"$\lambda$'s effect on side effects")
    plt.savefig('../report/figures/static_8x8_results_side_effects.png', bbox_inches='tight')

    fig, axis = plt.subplots(figsize=(WIDTH*cm,HEIGHT*cm), dpi=DPI)
    sns.lineplot(data=df, x='time_step', y='avg_obj', hue='lambda', palette=palette, legend=True)
    plt.xlabel('Timestep')
    plt.ylabel('Objective reward')
    plt.suptitle(r"$\lambda$'s effect on objective reward")
    plt.legend(title=r'$\lambda$', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('../report/figures/static_8x8_results_avg_obj.png', bbox_inches='tight')
