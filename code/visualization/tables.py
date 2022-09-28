import json
from collections import defaultdict
import os

import numpy as np
import pandas as pd

def save_table(dir):
    model = dir.split('/')[-1]
    df = summarize_data(dir)
    df.columns = ['$\lambda$', 'Length', 'Side effect', 'Objective']
    df.to_latex(buf=f'../report/tables/{model}.tex',
                         index=False, escape=False,
                         column_format='c|l|l|l')

def summarize_data(dir, top=30):

    # read data
    df = pd.read_csv(f'{dir}/df.csv')

    # extract last training time steps
    df_top = df[df.time_step > top]

    data = defaultdict(list)

    lambda_range = load_lambda_range(dir)
    for lmbda in lambda_range:
        df_top_lmbda = df_top[df_top['lambda'] == lmbda]

        data['lmbda'].append(lmbda)

        mean_len = df_top_lmbda['avg_len'].mean().round(2)
        std_len = df_top_lmbda['avg_len'].std().round(2)
        mean_side_effect = df_top_lmbda['avg_side_effects'].mean().round(2)
        std_side_effect = df_top_lmbda['avg_side_effects'].std().round(2)
        mean_obj = df_top_lmbda['avg_obj'].mean().round(2)
        std_obj = df_top_lmbda['avg_obj'].std().round(2)

        data['Length'].append(f'{mean_len} $\pm$ ({std_len})')
        data['Side effect'].append(f'{mean_side_effect} $\pm$ ({std_side_effect})')
        data['Objective'].append(f'{mean_obj} $\pm$ ({std_obj})')

    return pd.DataFrame(data)

def load_lambda_range(dir):
    with open(f'{dir}/params.json') as json_file:
        data = json.load(json_file)['aux_params']
    return data['lambda_range']


