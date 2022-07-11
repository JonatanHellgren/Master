import torch

from training import FeedForwardNN
from environment import MDP, EnvParams 
from training import PPO

""" TODO """
# save best model
# Add aux rews
# Train manager
# add manager aux

if __name__ == "__main__":
    env_params = EnvParams()
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

    ppo = PPO(mdp, actor, critic, device)
    ppo.learn(100, 1e4, 'test_training')
