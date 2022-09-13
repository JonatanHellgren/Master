import matplotlib.pyplot as plt

from environment import MDP, EnvParams
from visualization import plot_grid

def run_environment():
    env_params = EnvParams(
            (4,4), # size
            9,      # n_foods
            3,       # n_food_types
            100,     # n_test
            True)
    mdp = MDP(env_params)
    grid = mdp.reset()
    _, max_x, max_y = grid.shape
    fig, axis = plt.subplots(figsize=(1,1))

    done = False
    while not done:

        # add side effects and objectives to mdp.info and teturn with .step
        """
        plot_grid_lines(axis, max_x, max_y)
        plot_objects(grid, axis, max_x, max_y)
        fig.show()
        """
        plot_grid(grid, 'tmp.png')

        action = input('Action: ')
        action = int(action)
        grid, reward, done, info = mdp.step(action)

        axis.clear()
        if done:
            plot_grid(axis, max_x, max_y)
            plot_objects(grid, axis, max_x, max_y)
            fig.show()

if __name__ == "__main__":
    run_environment()
