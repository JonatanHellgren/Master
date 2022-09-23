from typing import NamedTuple

class TrainParameters(NamedTuple):
    """
    Object for storing training parameters
    """
    # Parameter                 Type  # Example
    timesteps_per_batch       : int   # 500
    max_timesteps_per_episode : int   # 500
    gamma                     : float # 0.95
    n_updates_per_iteration   : int   # 3
    clip                      : float # 0.1
    actor_lr                  : float # 1e-4
    critic_lr                 : float # 7e-4
    manager_lr                : float # 1e-4
    n_conv                    : int   # 64
    hidden_dim                : int   # 1024
