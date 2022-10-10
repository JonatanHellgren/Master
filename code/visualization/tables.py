import json
from collections import defaultdict
import os

import numpy as np
import pandas as pd

def save_table_extra(dir):
    model = dir.split('/')[-1]
    df = summarize_data_mean_std_and_rel_decrease(dir)
    df.columns = ['$\lambda$', 'mean', 'decrease', 'mean', 'decrease']
    df.to_latex(buf=f'../report/tables/{model}.tex',
                         index=False, escape=False,
                         column_format='c||c|c||c|c')
def save_table(dir):
    model = dir.split('/')[-1]
    df = summarize_data_rel_decrease(dir)
    df.columns = ['$\lambda$', 'Length', 'Side effect', 'Objective']
    df.to_latex(buf=f'../report/tables/{model}.tex',
                         index=False, escape=False,
                         column_format='c|l|l|l')

def summarize_data_mean_std_and_rel_decrease(dir, top=30):

    # read data
    df = pd.read_csv(f'{dir}/df.csv')

    # extract last training time steps
    df_top = df[df.time_step > top]

    data = defaultdict(list)

    lambda_range = load_lambda_range(dir)
    for lmbda in lambda_range:
        df_top_lmbda = df_top[df_top['lambda'] == lmbda]

        data['lmbda'].append(lmbda)

        # mean_len = df_top_lmbda['avg_len'].mean().round(2)
        # std_len = df_top_lmbda['avg_len'].std().round(2)
        mean_side_effect = df_top_lmbda['avg_side_effects'].mean().round(2)
        # std_side_effect = df_top_lmbda['avg_side_effects'].std().round(2)
        mean_obj = df_top_lmbda['avg_obj'].mean().round(2)
        # std_obj = df_top_lmbda['avg_obj'].std().round(2)

        if lmbda == lambda_range[0]:
            # base_len = mean_len
            base_side_effect = mean_side_effect
            base_obj = mean_obj

        # rel_increase_len = _rel_increase(base_len, mean_len)
        rel_increase_side_effect = _rel_increase(base_side_effect, mean_side_effect)
        rel_increase_obj = _rel_increase(base_obj, mean_obj)

        # data['Length'].append(f'{mean_len} $\pm$ ({std_len})')
        data['mean_side_effect'].append(mean_side_effect)
        # data['std_side_effect'].append(std_side_effect)
        data['rel_decrease_side_effect'].append(f'{abs(rel_increase_side_effect)}\%')

        data['mean_obj'].append(mean_obj)
        # data['std_obj'].append(std_obj)
        data['rel_decrease_obj'].append(f'{abs(rel_increase_obj)}\%')

    return pd.DataFrame(data)

def summarize_data_rel_decrease(dir, top=30):

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
        mean_side_effect = df_top_lmbda['avg_side_effects'].mean().round(2)
        mean_obj = df_top_lmbda['avg_obj'].mean().round(2)

        if lmbda == lambda_range[0]:
            base_len = mean_len
            base_side_effect = mean_side_effect
            base_obj = mean_obj

        rel_increase_len = _rel_increase(base_len, mean_len)
        rel_increase_side_effect = _rel_increase(base_side_effect, mean_side_effect)
        rel_increase_obj = _rel_increase(base_obj, mean_obj)

        data['Length'].append(f'{mean_len} ({rel_increase_len}\%)')
        data['Side effect'].append(f'{mean_side_effect} ({rel_increase_side_effect}\%)')
        data['Objective'].append(f'{mean_obj} ({rel_increase_obj}\%)')

    return pd.DataFrame(data)

def _rel_increase(old_value, new_value):
    change_factor = (new_value - old_value) / old_value
    percentage = round(100 * change_factor, 1)
    return percentage


def summarize_data_mean_std(dir, top=30):

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


