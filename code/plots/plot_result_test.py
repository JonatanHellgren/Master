import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

palette = sns.color_palette("colorblind")
col_ind = [0, 1, 2, 4, 8]
palette = [palette[i] for i in col_ind]

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)
plt.rcParams['text.usetex'] = True
"""
d = {'lambda': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
     'side_effect': [1, 2, 1.5, 1.25, 1.3, 1.4,
                     1, 0.5, 0.7, 0.5, 0.6, 0.6],
     'time_step': [1, 1, 2, 2, 3, 3,
                   1, 1, 2, 2, 3, 3]}
df = pd.DataFrame(d)
"""

df = pd.read_csv('models/static_8x8/df10.csv')
df = df.reset_index()
# df.critic_loss = df.critic_loss.apply(float)

height = 6
width = 4
dpi = 400 

# df = df[0:400]
cm = 1/2.54  # centimeters in inches
fig, axis = plt.subplots(figsize=(height*cm,width*cm), dpi=dpi)
sns.lineplot(data=df, x='time_step', y='avg_side_effects', hue='lambda', palette=palette, legend=False)
plt.xlabel('Timestep')
plt.ylabel('Side effect')
plt.suptitle(r"$\lambda$'s effect on side effects")
plt.savefig('../report/figures/static_8x8_results_side_effects.png', bbox_inches='tight')

fig, axis = plt.subplots(figsize=(height*cm,width*cm), dpi=dpi)
sns.lineplot(data=df, x='time_step', y='avg_obj', hue='lambda', palette=palette, legend=True)
plt.xlabel('Timestep')
plt.ylabel('Objective reward')
plt.suptitle(r"$\lambda$'s effect on objective reward")
plt.legend(title=r'$\lambda$', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('../report/figures/static_8x8_results_avg_obj.png', bbox_inches='tight')

"""
fig, axis = plt.subplots(figsize=(height*cm,width*cm), dpi=dpi)
sns.lineplot(data=df, x='time_step', y='avg_len', hue='lambda', palette=palette)
plt.xlabel('Timestep')
plt.ylabel('Mean length')
plt.suptitle(r"$\lambda$'s effect on task length")
plt.savefig('../report/figures/static_8x8_results_avg_len.png', bbox_inches='tight')
"""
# change colors
# rolling average?

# Higher critici loss with higher lambda. -> Harder for critic to estimate
# Need RNN critic, to estimate RNN managers output? 

# How can I notice when a agent does not learn? And perhaps reset it? 
# Or maybe more radomness at start?

