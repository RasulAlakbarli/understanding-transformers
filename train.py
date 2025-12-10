import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from transformers.bpe_tokenizer import BPE_Tokenizer
from transformers.dataset import Transformer_DS

# Read the textfile
full_text = open("/Users/macbookpro/Desktop/Transformers/data/quotes.txt", "r").readlines()

# Train the tokenizer on our data
tok_epochs = 5000
tokenizer = BPE_Tokenizer(text=" ".join(full_text))
tokenizer.fit(tok_epochs)

# Define dataset
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    return {"input_ids": input_ids, "labels": labels}

train_dataset = Transformer_DS(tokens=tokenizer.text_tokens, block_size=128)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn) # Inputs of shape [batch_size, block_size]

# Define the model
...

