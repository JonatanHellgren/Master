"""
All feed forward networks
"""
import pdb

import torch
from torch import nn, tensor, float, unsqueeze, flatten
from torch.nn import GRU
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np

class RecurrentNN(nn.Module):
    """
    Feed forward convolutional network.

    X should be on the shape: (batch, food_type, x_cord, y_cord)
    """

    def __init__(self, in_dim, n_conv, hidden_dim, out_dim, device, softmax=False, kernel_size=(3,3)):
        super(RecurrentNN, self).__init__()
        self.softmax = softmax
        self.device = device

        # Collecting dimentions
        n_z, n_x, n_y = in_dim
        flat_dim = (n_x - kernel_size[0] + 1) * (n_y - kernel_size[1] + 1) * n_conv * 2
        self.flat_dim = flat_dim
        self.hidden_dim = hidden_dim

        # Convolutional layers
        self.conv1 = nn.Conv2d(n_z, n_conv, kernel_size)
        self.conv2 = nn.Conv2d(n_conv, 2*n_conv, kernel_size, padding=1)

        self.rnn = nn.GRU(flat_dim, hidden_dim, 3)

        # Linear layers
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, obs, batch_lens):
        """
        A forward pass for the network
        """
        if isinstance(obs, np.ndarray):
            obs = tensor(obs, dtype=float)
            if len(obs.size()) == 3:
                obs = unsqueeze(obs, dim=0)

        obs = obs.to(self.device)
        # print(obs.size())

        feature_map1 = F.relu(self.conv1(obs))
        feature_map2 = F.relu(self.conv2(feature_map1))

        # pdb.set_trace()
        flat = flatten(feature_map2, start_dim=1) # (batch, feature)

        sequences = _split_batch_to_seq(flat, batch_lens)
        sequences_padded = pad_sequence(sequences, batch_first=True)
        sequences_padded = pack_padded_sequence(
                sequences_padded, batch_lens, enforce_sorted=False, batch_first=True)

        out = self.rnn(sequences_padded)
        out_unpacked = pad_packed_sequence(out[0], batch_first=True)

        combined_sequence = _padded_to_sequence(out_unpacked, self.hidden_dim) 
        combined_sequence = combined_sequence.to(self.device)

        activation1 = F.relu(self.layer1(combined_sequence))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        # Returning output with softmax activation, used for the actor network
        if self.softmax:
            return F.softmax(output, dim=1)

        return output

def _padded_to_sequence(out_unpacked, flat_dim):
    # print(out_unpacked[1])
    # print(out_unpacked[1].sum())
    # print(flat_dim)
    combined_sequence = torch.zeros(out_unpacked[1].sum(), flat_dim)
    current_ind = 0
    for ind, batch_len in enumerate(out_unpacked[1]):
        seq = out_unpacked[0][ind][0:batch_len]
        # print(seq)
        # print(seq.size())
        seq = torch.squeeze(seq,1)
        combined_sequence[current_ind:current_ind+batch_len, :] = seq
        current_ind += batch_len
    return combined_sequence

def _split_batch_to_seq(features, batch_lens):
    sequences = []
    current_ind = 0
    for batch_len in batch_lens:
        sequences.append(features[current_ind:current_ind+batch_len])
        current_ind += batch_len

    return sequences

class FeedForwardNN(nn.Module):
    """
    Feed forward convolutional network.

    X should be on the shape: (batch, food_type, x_cord, y_cord)
    """

    def __init__(self, in_dim, n_conv, hidden_dim, out_dim, device, softmax=False, kernel_size=(3,3)):
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
            obs = tensor(obs, dtype=float)
            if len(obs.size()) == 3:
                obs = unsqueeze(obs, dim=0)

        obs = obs.to(self.device)
        # print(obs.size())

        feature_map1 = F.relu(self.conv1(obs))
        feature_map2 = F.relu(self.conv2(feature_map1))

        flat = flatten(feature_map2, start_dim=1) # (batch, feature)

        activation1 = F.relu(self.layer1(flat))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        # Returning output with softmax activation, used for the actor network
        if self.softmax:
            return F.softmax(output, dim=1)

        return output
