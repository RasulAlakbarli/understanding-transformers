import torch
from torch.utils.data import Dataset

class Transformer_DS(Dataset):
    """ 
    Transformer Dataset for token sequences.
    
    Args:
        tokens (list[int]): List of token ids.
        block_size (int): Size of each input sequence block.
    """
    def __init__(self, tokens: list[int], block_size=128):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size
    def __len__(self):
        return len(self.tokens) - self.block_size
    def __getitem__(self, idx):
        x = self.tokens[idx: idx+self.block_size]
        y = self.tokens[idx+1: idx+self.block_size+1]
        return {"input_ids": x, "labels": y}