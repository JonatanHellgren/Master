import json
import argparse

import torch
import numpy as np
import pandas as pd

from training import TrainParameters, PPO, ManagerTrainer
from environment import MDP, EnvParams
from networks import FeedForwardNN, Agent

def main():
    """
    Main is the function that sets everything going. It begins with parsing the arguments,
    then it trains a agent and manager if necessary, lastly it executes the testing of
    auxiliary values.
    """

    # Create the parser
    parser = argparse.ArgumentParser(description='List the content of a folder')

    # Add the arguments
    parser.add_argument('-p', '--path', type=str, help='the path to model folder')
    parser.add_argument('-n', dest='new', action='store_true',
            help='pass to train new agent and manager')
    parser.set_defaults(new=False)
    args = parser.parse_args()
    model_dir = args.path

    # train new agent and manager if new instance
    if args.new:
        train_agent(model_dir)
        train_manager(model_dir)

    # exectute tests with impact measurement
    impact_measurement_test(model_dir)

def train_agent(model_dir):
    """
    Function for training the agent. Loads the pparameters from the model directory and
    saves the best agent from the training in the same directory.
    """
    environment, train_params, device = initialize_environment(model_dir)

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
    ppo.train(100, 1e4, model_dir)

def train_manager(model_dir):
    """
    Trains the manager using the parameters in the model directory. Here the best manager
    with the lowest test loss is saved in the same directory.
    """
    environment, train_params, device = initialize_environment(model_dir)

    # inializing actor
    actor = FeedForwardNN(environment.obs_size, train_params.n_conv,
        train_params.hidden_dim, environment.n_actions, device, softmax=True).to(device)
    # loading pre-trained critic
    actor.load_state_dict(torch.load(f'./{model_dir}/best_actor.model',
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
    manager_trainer.train(500, 1e4, model_dir)

def impact_measurement_test(model_dir):
    """
    Function for testing the impact measurement with different \lambda values.
    Stores a dataframe containing all the results in the model directory.
    """
    environment, train_params, device = initialize_environment(model_dir)

    # inializing manager
    manager = FeedForwardNN(environment.obs_size, train_params.n_conv,
        train_params.hidden_dim, 1, device).to(device)
    # and pre-trained manager
    manager.load_state_dict(torch.load(f'./{model_dir}/best_manager.model',
        map_location=torch.device(device)))

    n_runs, n_epochs, lambda_range, out_file_name = _load_aux_params(model_dir)

    results_df = pd.DataFrame()
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
            results_df = pd.concat([results_df, pd.DataFrame(log)])

    results_df = results_df.reset_index()
    results_df.to_csv(f'{model_dir}/{out_file_name}')

def initialize_environment(model_dir):
    """
    This function sets up the environment.
    Returns the environment, the loaded training parameters and which device loaded on.
    """
    # find what device is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    environment = _load_env(model_dir)
    train_params = _load_train_params(model_dir)

    return environment, train_params, device

def _load_env(model_dir):
    env_params = _load_env_params(model_dir)
    environment = MDP(env_params)
    return environment

def _load_env_params(model_dir):
    with open(f'{model_dir}/params.json') as json_file:
        data = json.load(json_file)['env_params']

    env_params = EnvParams(
            (data['size'], data['size']), # always square grid
             data['n_foods'],
             data['n_food_types'],
             data['objective'],
             data['n_test'],
             data['is_stochastic'],
             data['is_pomdp'])

    return env_params


def _load_train_params(model_dir):
    with open(f'{model_dir}/params.json') as json_file:
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

def _load_aux_params(model_dir):
    with open(f'{model_dir}/params.json') as json_file:
        data = json.load(json_file)['aux_params']

    return data['n_runs'], data['n_epochs'], data['lambda_range'], data['out_file_name']

if __name__ == "__main__":
    main()
