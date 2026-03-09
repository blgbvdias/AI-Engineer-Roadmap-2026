import torch

def get_batch(data_tensor, seq_len, batch_size):
    max_index = len(data_tensor) - seq_len - 1
    ix = torch.randint(0, max_index, (batch_size,))
    x = torch.stack([data_tensor[i:i+seq_len] for i in ix])
    y = torch.stack([data_tensor[i+1:i+seq_len+1] for i in ix])
    # x, y = x.to(device='cuda'), y.to(device='cuda')
    return x, y
