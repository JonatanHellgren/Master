import pickle
import json
import argparse

import torch
import numpy as np
import pandas as pd

from training import TrainParameters, PPO, ManagerTrainer
from environment import MDP, EnvParams 
from networks import FeedForwardNN, Agent

# DIR = 'models/testy'

def main():
    # Create the parser
    my_parser = argparse.ArgumentParser(description='List the content of a folder')

    # Add the arguments
    my_parser.add_argument('-p',
                           '--path',
                           type=str,
                           help='the path to list')

    # Execute the parse_args() method
    args = my_parser.parse_args()
    
    dir = args.path
    train_agent(dir)
    train_manager(dir)
    train_aux(dir)

def train_agent(DIR):

    environment, train_params, device = initialize_environment(DIR)

    # initialize agent
    actor = FeedForwardNN(environment.obs_size, train_params.n_conv, 
        train_params.hidden_dim, environment.n_actions, device, softmax=True).to(device)
    # and critc
    critic = FeedForwardNN(environment.obs_size, train_params.n_conv,
        train_params.hidden_dim, 1, device).to(device)
    # now manager
    agent = Agent(actor, critic, train_params)

    # train agent
    ppo = PPO(environment, agent, device, train_params)
    ppo.train(10, 1e4, DIR)

def train_manager(DIR):

    environment, train_params, device = initialize_environment(DIR)

    # inializing actor
    actor = FeedForwardNN(environment.obs_size, train_params.n_conv, 
        train_params.hidden_dim, environment.n_actions, device, softmax=True).to(device)
    # loading pre-trained critic
    actor.load_state_dict(torch.load(f'./{DIR}/best_actor.model',
        map_location=torch.device(device)))

    # initialize critic
    critic = FeedForwardNN(environment.obs_size, train_params.n_conv,
        train_params.hidden_dim, 1, device).to(device)
    # and manager
    manager = FeedForwardNN(environment.obs_size, train_params.n_conv,
        train_params.hidden_dim, 1, device).to(device)

    # agent with manager
    agent = Agent(actor, critic, train_params, manager)

    # train manager
    manager_trainer = ManagerTrainer(environment, agent, device, train_params)
    manager_trainer.train(5, 1e4, DIR)

def train_aux(DIR):

    environment, train_params, device = initialize_environment(DIR)

    # inializing manager
    manager = FeedForwardNN(environment.obs_size, train_params.n_conv,
        train_params.hidden_dim, 1, device).to(device)
    # and pre-trained manager
    manager.load_state_dict(torch.load(f'./{DIR}/best_manager.model',
        map_location=torch.device(device)))
    
    n_runs, n_epochs, lambda_range, out_file_name = load_aux_params(DIR)
    
    df = pd.DataFrame()
    for run in range(n_runs):
        for lmbda in lambda_range:
            print(f"Run = {run}\nLambda = {lmbda}")

            # initialize agent
            actor = FeedForwardNN(environment.obs_size, train_params.n_conv, 
                train_params.hidden_dim, environment.n_actions, device, softmax=True).to(device)
            # and critc
            critic = FeedForwardNN(environment.obs_size, train_params.n_conv,
                train_params.hidden_dim, 1, device).to(device)

            agent = Agent(actor, critic, train_params, manager)

            ppo_aux = PPO(environment, agent, device, train_params,
                    lmbda=lmbda, save_model=False)
            ppo_aux.train(n_epochs, 1e4, '.')
            log = ppo_aux.logging
            log['run'] = np.ones(n_epochs) * run
            log['lambda'] = np.ones(n_epochs) * lmbda
            log['time_step'] = list(range(1, n_epochs+1))
            df = pd.concat([df, pd.DataFrame(log)])

    df = df.reset_index()
    df.to_csv(f'{DIR}/{out_file_name}')

def initialize_environment(DIR):

    # find what device is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    environment = load_env(DIR)
    train_params = load_train_params(DIR)

    return environment, train_params, device

def load_env_params(DIR):
    with open(f'{DIR}/params.json') as json_file:
        data = json.load(json_file)['env_params']

    env_params = EnvParams(
            (data['size'], data['size']), # always square grid
             data['n_foods'],
             data['n_food_types'],
             data['n_test'],
             data['is_stochastic'],
             data['is_pomdp'])

    return env_params

def load_env(DIR):
    env_params = load_env_params(DIR)
    environment = MDP(env_params)
    return environment

def load_train_params(DIR):
    with open(f'{DIR}/params.json') as json_file:
        data = json.load(json_file)['train_params']

    train_params = TrainParameters(
            data['timesteps_per_batch'],
            data['max_timesteps_per_episode'],
            data['gamma'],
            data['n_updates_per_iteration'],
            data['clip'],
            data['actor_lr'],
            data['critic_lr'],
            data['manager_lr'],
            data['n_conv'],
            data['hidden_dim'])

    return train_params

def load_aux_params(DIR):
    with open(f'{DIR}/params.json') as json_file:
        data = json.load(json_file)['aux_params']

    return data['n_runs'], data['n_epochs'], data['lambda_range'], data['out_file_name']

if __name__ == "__main__":
    main()
