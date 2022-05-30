"""
Parameters for networks
"""
conv_size = (3, 3)
n_conv = 16
hidden_dim = 128

"""
hyperparameters for training
"""
timesteps_per_batch = 4800
max_timesteps_per_episide = 100
total_timesteps = 9600
n_updates_per_iteration = 10
n_epochs = 10
kick_in = 500
clip = 0.2
lr = 1e-3

#= 73s - 75s =#
# but ~21s on gpu
