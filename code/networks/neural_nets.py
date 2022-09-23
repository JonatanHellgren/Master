"""
All feed forward networks
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    """
    Feed forward convolutional network.

    X should be on the shape: (batch, food_type, x_cord, y_cord)
    """

    def __init__(self, in_dim, n_conv, hidden_dim, out_dim, device,
                 softmax=False, kernel_size=(3,3)):
        super(FeedForwardNN, self).__init__()
        self.softmax = softmax
        self.device = device

        # Collecting dimentions
        n_z, n_x, n_y = in_dim
        flat_dim = (n_x - kernel_size[0] + 1) * (n_y - kernel_size[1] + 1) * n_conv * 2

        # Convolutional layers
        self.conv1 = nn.Conv2d(n_z, n_conv, kernel_size)
        self.conv2 = nn.Conv2d(n_conv, 2*n_conv, kernel_size, padding=1)

        # Linear layers
        self.layer1 = nn.Linear(flat_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, obs):
        """
        A forward pass for the network
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
            if len(obs.size()) == 3:
                obs = torch.unsqueeze(obs, dim=0)

        obs = obs.to(self.device)
        # print(obs.size())

        feature_map1 = F.relu(self.conv1(obs))
        feature_map2 = F.relu(self.conv2(feature_map1))

        flat = torch.flatten(feature_map2, start_dim=1) # (batch, feature)

        activation1 = F.relu(self.layer1(flat))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        # Returning output with softmax activation, used for the actor network
        if self.softmax:
            return F.softmax(output, dim=1)

        return output
