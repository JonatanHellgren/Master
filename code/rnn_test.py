import torch
from torch.nn import GRU
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

batch_lens = [10, 11, 14, 14, 11, 12, 10, 23]

# batch_obs
# torch.Size([105, 4, 5, 5])

# batch_rtgs
# torch.Size([105])

batch_obs = torch.ones([105, 10])

def split_batch_to_seq(features, batch_lens):
    sequences = []
    current_ind = 0
    for batch_len in batch_lens:
        sequences.append(batch_obs[current_ind:current_ind+batch_len])
        current_ind += batch_len

    return sequences

sequences = split_batch_to_seq(batch_obs, batch_lens)
sequences_padded = pad_sequence(sequences, batch_first=True)

sequences_padded = pack_padded_sequence(
        sequences_padded, batch_lens, enforce_sorted=False, batch_first=True)

rnn = GRU(10, 10, 1)
out = rnn(sequences_padded)
out_unpacked = pad_packed_sequence(out[0], batch_first=True)

# Return detached rtgs from manager

def padded_to_sequence(out_unpacked):
    rtgs = torch.zeros(out_unpacked[1].sum(), 10)
    current_ind = 0
    for ind, batch_len in enumerate(out_unpacked[1]):
        seq = out_unpacked[0][ind][0:batch_len]
        seq = torch.squeeze(seq,1)
        rtgs[current_ind:current_ind+batch_len, :] = seq
        current_ind += batch_len
    return rtgs

rtgs = padded_to_sequence(out_unpacked)


    
