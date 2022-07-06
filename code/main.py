from environment import MDP, EnvParams
from training import PPO


import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

env_params = EnvParams()
mdp = MDP(env_params)
ppo = PPO(mdp, 64, 1024)
ppo.learn(1000, 1e4)
"""
batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = ppo.rollout()
env.reset()
env.step(1)
# env.step(3)
"""