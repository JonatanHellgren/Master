import pickle

import torch

from training import TrainParameters, ManagerTrainer
from environment import MDP, EnvParams 
from networks import FeedForwardNN, Agent

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

    train_parameters = TrainParameters(
            500,  # timesteps_per_batch 
            500,  # max_timesteps_per_episode 
            0.95, # gamma 
            3,    # n_updates_per_iteration 
            0.1,  # clip 
            1e-4, # actor_lr 
            7e-4, # critic_lr
            1e-4) # manager_lr

    actor = FeedForwardNN(obs_dim, n_conv, hidden_dim, act_dim, device, softmax=True).to(device)
    actor.load_state_dict(torch.load('./models/static/best_actor', map_location=torch.device('cpu')))

    critic = FeedForwardNN(obs_dim, n_conv, hidden_dim, 1, device).to(device)
    manager = FeedForwardNN(obs_dim, n_conv, hidden_dim, 1, device).to(device)

    agent = Agent(actor, critic, train_parameters, manager)

    manager_trainer = ManagerTrainer(mdp, agent, device, train_parameters)
    manager_trainer.train(100, 1e4, DIR)



