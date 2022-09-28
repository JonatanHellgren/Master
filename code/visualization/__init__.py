import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt

palette = sns.color_palette("colorblind")
BLUE, RED, GREEN = palette[0:3]
col_ind = [0, 1, 2, 4, 6, 7, 8]
palette = [palette[i] for i in col_ind]

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['text.usetex'] = True

from .visualization import plot_grid, print_grid
from .tables import save_table
from .training_history import plot_training_history

