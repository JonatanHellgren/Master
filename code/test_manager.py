import torch
from numpy import random

from training import TrainParameters
from environment import MDP, EnvParams 
from networks import FeedForwardNN, Agent
from training.training_utils import rollout

DIR = 'models/static_4x4'

def test_augmentation():
    """
    Tests if the manager outputs similar expected utilities when augmenting 
    the environment
    """
    # setup environment
    env_params = EnvParams(
            (4, 4),  # size
            9,      # n_foods
            3,       # n_food_types
            100)     # n_test
    mdp = MDP(env_params, pomdp=False)

    obs_dim = mdp.obs_size
    act_dim = mdp.n_actions

    # Find what device is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loading manager for 4x4 environment
    n_conv = 64
    hidden_dim = 1024
    manager = FeedForwardNN(obs_dim, n_conv, hidden_dim, 1, device).to(device)
    manager.load_state_dict(torch.load(f'./{DIR}/best_manager', map_location=torch.device(device)))

    random.seed(1)
    grid = mdp.reset()

    # Get expected value from manager using the original grid
    v = manager(grid)

    # Compare expected values between original and augmented grids 
    # Check if they are within a range of 0.25
    v_aug_1 = manager(grid[[0,2,1,3],:,:])
    assert torch.abs(v - v_aug_1) < 0.25

    v_aug_2 = manager(grid[[0,2,3,1],:,:])
    assert torch.abs(v - v_aug_2) < 0.25

    v_aug_3 = manager(grid[[0,3,2,1],:,:])
    assert torch.abs(v - v_aug_3) < 0.25

def test_manager_decrease():
    env_params = EnvParams(
            (4, 4),  # size
            9,      # n_foods
            3,       # n_food_types
            100)     # n_test
    mdp = MDP(env_params, pomdp=False)

    obs_dim = mdp.obs_size
    act_dim = mdp.n_actions

    # Find what device is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_parameters = TrainParameters(
            100,  # timesteps_per_batch 
            500,  # max_timesteps_per_episode 
            0.95, # gamma 
            3,    # n_updates_per_iteration 
            0.1,  # clip 
            1e-4, # actor_lr 
            7e-4, # critic_lr 
            1e-4, # manager_lr
            0.1)    # lmbda


    n_conv = 64
    hidden_dim = 1024

    manager = FeedForwardNN(obs_dim, n_conv, hidden_dim, 1, device).to(device)
    manager.load_state_dict(torch.load(f'./models/static/best_manager',\
                            map_location=torch.device(device)))
    actor = FeedForwardNN(obs_dim, n_conv, hidden_dim, act_dim, device, softmax=True).to(device)
    actor.load_state_dict(torch.load(f'./{DIR}/best_actor', map_location=torch.device(device)))
    # and critic
    critic = FeedForwardNN(obs_dim, n_conv, hidden_dim, 1, device).to(device)

    agent = Agent(actor, critic, train_parameters)

    batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = \
            rollout(agent, train_parameters, mdp)
