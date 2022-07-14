from typing import NamedTuple

class TrainParameters(NamedTuple):
    # Parameter                 Type # Example
    timesteps_per_batch       : int  # 500
    max_timesteps_per_episode : int  # 500
    gamma                     : int  # 0.95
    n_updates_per_iteration   : int  # 3
    clip                      : int  # 0.1
    actor_lr                  : int  # 1e-4
    critic_lr                 : int  # 7e-4
    manager_lr                : int  # 1e-4
