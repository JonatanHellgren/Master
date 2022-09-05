import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
d = {'lambda': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
     'side_effect': [1, 2, 1.5, 1.25, 1.3, 1.4,
                     1, 0.5, 0.7, 0.5, 0.6, 0.6],
     'time_step': [1, 1, 2, 2, 3, 3,
                   1, 1, 2, 2, 3, 3]}
df = pd.DataFrame(d)
"""

df = df.reset_index()
sns.lineplot(data=df, x='time_step', y='avg_obj', hue='lambda')
plt.show()

# change colors
# rolling average?

