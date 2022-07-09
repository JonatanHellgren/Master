from environment import MDP, EnvParams 
from training import PPO

""" TODO """
# Sep MDP and state
# Add aux rews
# Train manager
# add manager aux

if __name__ == "__main__":
    env_params = EnvParams()
    mdp = MDP(env_params, pomdp=True)
    ppo = PPO(mdp, 64, 1024)
    ppo.learn(1000, 1e4)
