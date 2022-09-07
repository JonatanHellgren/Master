import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

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

cm = 1/2.54  # centimeters in inches
fig, axis = plt.subplots(figsize=(14*cm,10*cm), dpi=100)
sns.lineplot(data=df[df.run!=4], x='time_step', y='avg_side_effects', hue='lambda')
plt.xlabel('Timestep')
plt.ylabel('Mean Side Effect')
# plt.suptitle('\$\lambda\$')
plt.savefig('static_8x8_results_side_effects.png')

# change colors
# rolling average?

# Higher critici loss with higher lambda. -> Harder for critic to estimate
# Need RNN critic, to estimate RNN managers output? 

# How can I notice when a agent does not learn? And perhaps reset it? 
# Or maybe more radomness at start?

