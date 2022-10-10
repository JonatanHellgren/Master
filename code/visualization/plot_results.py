from collections import defaultdict
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc


WIDTH = 8
HEIGHT = 5
DPI = 400 
cm = 1/2.54  # centimeters in inches

palette = sns.color_palette("colorblind")
# col_ind = [0, 1, 2, 4, 6, 7, 8]
# palette = [palette[i] for i in col_ind]

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['text.usetex'] = True

def summarize_data(DIR, sup_title, top=30):

    fig, axis = plt.subplots(figsize=(WIDTH*cm,HEIGHT*cm), dpi=DPI)

    # read data
    df = pd.read_csv(f'{DIR}/df.csv')
    df_stochastic = pd.read_csv(f'{DIR}_stochastic/df.csv')

    df_mean = get_mean_data(df, DIR)
    df_stochastic_mean = get_mean_data(df_stochastic, DIR)

    sns.lineplot(df_mean, x='lmbda', y='Objective', color=palette[3], label='non-stochastic objective')
    sns.lineplot(df_mean, x='lmbda', y='Side effect', color=palette[3], linestyle='--', label='non-stochastic side effect')
    sns.lineplot(df_stochastic_mean, x='lmbda', y='Objective', color=palette[8], label='stochastic objective')
    sns.lineplot(df_stochastic_mean, x='lmbda', y='Side effect', color=palette[8], linestyle='--', label='stochastic side effect')

    model_name = DIR.split('/')[-1]
    plt.suptitle(sup_title)
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Consumed food object')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f'../report/figures/{model_name}_result_plot.png', bbox_inches='tight')
    # plt.show()


def get_mean_data(df, DIR, top=30):

    # extract last training time steps
    df_top = df[df.time_step > top]

    data = defaultdict(list)
    lambda_range = load_lambda_range(DIR)
    for lmbda in lambda_range:
        df_top_lmbda = df_top[df_top['lambda'] == lmbda]

        data['lmbda'].append(lmbda)

        mean_len = df_top_lmbda['avg_len'].mean()
        mean_side_effect = df_top_lmbda['avg_side_effects'].mean()
        mean_obj = df_top_lmbda['avg_obj'].mean()

        data['Length'].append(mean_len)
        data['Side effect'].append(mean_side_effect)
        data['Objective'].append(mean_obj)

    return pd.DataFrame(data)

def load_lambda_range(DIR):
    with open(f'{DIR}/params.json') as json_file:
        data = json.load(json_file)['aux_params']
    return data['lambda_range']


summarize_data('./models/pomdp_16x16', r'POMDP 16$\times$16')
summarize_data('./models/pomdp_8x8', r'POMDP 8$\times$8')
summarize_data('./models/static_8x8', r'MDP 8$\times$8')
