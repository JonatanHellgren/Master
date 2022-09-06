import pickle

import torch
import numpy as np
import pandas as pd

from training import TrainParameters, PPO, ManagerTrainer
from environment import MDP, EnvParams 
from networks import RecurrentNN, FeedForwardNN, Agent

DIR = 'models/pomdp_8x8'

env_params = EnvParams(
        (8, 8),  # size
        15,       # n_foods
        3,       # n_food_types
        100)     # n_test

def train_manager():
    mdp = MDP(env_params, pomdp=True)

    obs_dim = mdp.obs_size
    act_dim = mdp.n_actions

    # Find what device is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_conv = 64
    hidden_dim = 1024

    train_parameters = TrainParameters(
            100,  # timesteps_per_batch 
            500,  # max_timesteps_per_episode 
            0.95, # gamma 
            3,    # n_updates_per_iteration 
            0.1,  # clip 
            1e-4, # actor_lr 
            7e-4, # critic_lr 
            1e-4, # manager_lr
            0.1)  # lmbda

    # Initilize actor 
    actor = FeedForwardNN(obs_dim, n_conv, hidden_dim, act_dim, device, softmax=True).to(device)
    actor.load_state_dict(torch.load(f'./{DIR}/best_actor.model', map_location=torch.device(device)))
    critic = FeedForwardNN(obs_dim, n_conv, hidden_dim, 1, device).to(device)
    manager = RecurrentNN(obs_dim, n_conv, hidden_dim, 1, device).to(device)

    agent = Agent(actor, critic, train_parameters, manager, pomdp=True)

    manager_trainer = ManagerTrainer(mdp, agent, device, train_parameters)
    manager_trainer.train(500, 1e4, DIR)

def train_agent():
    mdp = MDP(env_params, pomdp=True)

    obs_dim = mdp.obs_size
    act_dim = mdp.n_actions

    # Find what device is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_conv = 64
    hidden_dim = 1024

    train_parameters = TrainParameters(
            100,  # timesteps_per_batch 
            500,  # max_timesteps_per_episode 
            0.95, # gamma 
            3,    # n_updates_per_iteration 
            0.1,  # clip 
            1e-4, # actor_lr 
            7e-4, # critic_lr 
            1e-4, # manager_lr
            0.1)  # lmbda

    # Initilize actor 
    actor = FeedForwardNN(obs_dim, n_conv, hidden_dim, act_dim, device, softmax=True).to(device)
    # and critic
    critic = FeedForwardNN(obs_dim, n_conv, hidden_dim, 1, device).to(device)

    agent = Agent(actor, critic, train_parameters)

    """
    Train agent
    """
    ppo = PPO(mdp, agent, device, train_parameters)
    ppo.train(100, 1e4, DIR)

# train_manager()
def train_aux():
    mdp = MDP(env_params, pomdp=True)

    obs_dim = mdp.obs_size
    act_dim = mdp.n_actions

    # Find what device is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_conv = 64
    hidden_dim = 1024

    manager = RecurrentNN(obs_dim, n_conv, hidden_dim, 1, device).to(device)
    manager.load_state_dict(torch.load(f'./{DIR}/best_manager.model', map_location=torch.device(device)))
    
    loggings = []
    n_runs = 10
    n_epochs = 100
    df = pd.DataFrame()
    for run in range(n_runs):
        for lmbda in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        # for lmbda in [0, 0.25, 0.5, 0.75, 1]:
            print(f"Run = {run}\nLambda = {lmbda}")
            train_parameters = TrainParameters(
                    100,  # timesteps_per_batch 
                    500,  # max_timesteps_per_episode 
                    0.95, # gamma 
                    3,    # n_updates_per_iteration 
                    0.1,  # clip 
                    1e-4, # actor_lr 
                    7e-4, # critic_lr 
                    1e-4, # manager_lr
                    lmbda)# lmbda

            actor = FeedForwardNN(obs_dim, n_conv, hidden_dim, act_dim, device, softmax=True).to(device)
            critic = FeedForwardNN(obs_dim, n_conv, hidden_dim, 1, device).to(device)
            agent = Agent(actor, critic, train_parameters, manager)

            ppo_aux = PPO(mdp, agent, device, train_parameters, use_aux=True)
            ppo_aux.train(n_epochs, 1e4, '.')
            log = ppo_aux.logging
            log['run'] = np.ones(n_epochs) * run
            log['lambda'] = np.ones(n_epochs) * lmbda
            log['time_step'] = list(range(1, n_epochs+1))
            df = pd.concat([df, pd.DataFrame(log)])

            # loggings.append(ppo_aux.logging)
    return df
