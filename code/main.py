import pickle

import torch

from training import FeedForwardNN, TrainParameters, PPO
from environment import MDP, EnvParams 

""" TODO """
# Train static
# Add aux rews
# Train manager using best actor
# add manager aux

DIR = 'models/static'

if __name__ == "__main__":
    env_params = EnvParams(
            (10,10), # size
            15,      # n_foods
            3,       # n_food_types
            100)     # n_test
    mdp = MDP(env_params, pomdp=True)

    obs_dim = mdp.obs_size
    act_dim = mdp.n_actions

    # Find what device is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_conv = 64
    hidden_dim = 1024

    # Initilize actor 
    actor = FeedForwardNN(obs_dim, n_conv, hidden_dim, act_dim, device, softmax=True).to(device)
    # and critic
    critic = FeedForwardNN(obs_dim, n_conv, hidden_dim, 1, device).to(device)
    train_parameters = TrainParameters(
            500,  # timesteps_per_batch 
            500,  # max_timesteps_per_episode 
            0.95, # gamma 
            3,    # n_updates_per_iteration 
            0.1,  # clip 
            1e-4, # actor_lr 
            7e-4) # critic_lr 

    ppo = PPO(mdp, actor, critic, device, train_parameters)
    ppo.learn(1000, 1e4, DIR)

    with open(f'{DIR}/logging.pickle', 'wb') as handle:
        pickle.dump(ppo.logging, handle, protocol=pickle.HIGHEST_PROTOCOL)
    """
    with open(f'{DIR}/logging.pickle', 'rb') as handle:
        unserialized_data = pickle.load(handle)
    """
