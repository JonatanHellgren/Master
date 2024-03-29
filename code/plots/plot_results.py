"""
"""
model_name = 'POMDP 8x8'
model = 'pomdp_8x8'

"""
model_name = 'POMDP 8x8 stochastic'
model = 'pomdp_8x8_stochastic'

model_name = 'MDP 8x8'
model = 'static_8x8'

model_name = 'MDP 8x8 stochastic'
model = 'static_8x8_stochastic'

model_name = "POMDP 16x16"
model = "pomdp_16x16"

model_name = "POMDP 16x16"
model = "pomdp_16x16_stochastic"
"""


DIR = f'../models/{model}'

def summarize_data(DIR, top=30):

    # read data
    df = pd.read_csv(f'{DIR}/df.csv')

    # extract last training time steps
    df_top = df[df.time_step > top]

    data = defaultdict(list)

    lambda_range = load_lambda_range(DIR)
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

def load_lambda_range(DIR):
    with open(f'{DIR}/params.json') as json_file:
        data = json.load(json_file)['aux_params']
    return data['lambda_range']

def save_table(df):
    df.columns = ['$\lambda$', 'Length', 'Side effect', 'Objective']
    df.to_latex(buf=f'../../report/tables/{model}.tex',
                         index=False, escape=False,
                         column_format='c|c|c|c')

df = summarize_data(DIR)
save_table(df)

