import math
import torch
import torch.nn as nn

from transformers.attention import MultiHeadAttention
from transformers.pos_encoding import SinusoidalPositionalEncoding

class EncoderBlock(nn.Module):
	"""
	Encoder Block
	"""
	def __init__(self, d_model: int):
		super().__init__()
		self.attn = MultiHeadAttention(d_model=d_model, n_heads=8)
		self.ff = nn.Sequential(
			nn.Linear(d_model, 4*d_model),
			nn.GELU(),
			nn.Linear(4*d_model, d_model),
		)
		self.dropout = nn.Dropout(0.1)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
  
	def forward(self, x):
		attn_out = self.attn(x) # [batch_size, N_seq, d_model]
		# Truns out that pre normalization is more stable than post normalization (more modern models use pre-norm)
		x = self.norm1(x + self.dropout(attn_out))
		ff_out = self.ff(x)
		x = self.norm2(x + self.dropout(ff_out))
  
		return x # [batch_size, N_seq, d_model]


class DecoderBlock(nn.Module):
	"""
	Decoder Block
	"""
	def __init__(self, d_model: int):
		super().__init__()
		self.attn_masked = MultiHeadAttention(d_model=d_model, n_heads=8)
		self.attn = MultiHeadAttention(d_model=d_model, n_heads=8)
		self.ff = nn.Sequential(
			nn.Linear(d_model, 4*d_model),
			nn.GELU(),
			nn.Linear(4*d_model, d_model),
		)
		self.dropout = nn.Dropout(0.1)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.norm3 = nn.LayerNorm(d_model)
  
	def forward(self, x, enc_kv_input: torch.Tensor = None):
		batch_size, seq_len, _ = x.size()
		mask = torch.tril(torch.ones(batch_size, 1, seq_len, seq_len)).to(x.device)
		masked_attn_out = self.attn_masked(x, mask=mask)
		# Truns out that pre normalization is more stable than post normalization (more modern models use pre-norm)
		x = self.norm1(x + self.dropout(masked_attn_out))
		if enc_kv_input is not None:
			attn_out = self.attn(x, kv_input=enc_kv_input)
			x = self.norm2(x + self.dropout(attn_out))
		ff_out = self.ff(x)
		x = self.norm3(x + self.dropout(ff_out))
  
		return x


class Transformer(nn.Module):
	"""
	Transformer Model from "Attention is All You Need" paper
	"""
	def __init__(self, d_model, N_stack, vocab_size):
		super().__init__()
		self.d_model = d_model
	
		self.enc_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
		self.dec_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
		self.pos_encoding = SinusoidalPositionalEncoding(d_model=d_model)
  
		self.encoder = nn.ModuleList([EncoderBlock(d_model) for _ in range(N_stack)])
		self.decoder = nn.ModuleList([DecoderBlock(d_model) for _ in range(N_stack)])
  
		self.linear = nn.Linear(d_model, vocab_size)
		self.softmax = nn.Softmax(dim=-1)
		
	def forward(self, x_enc, x_dec):
		# 1. Pass Inputs Through Embedding Layer
		x_enc = self.enc_embedding(x_enc) * math.sqrt(self.d_model)
		x_dec = self.dec_embedding(x_dec) * math.sqrt(self.d_model)

		# 2. Add Positional Encodings
		x_enc = self.pos_encoding(x_enc)
		x_dec = self.pos_encoding(x_dec)

		# 3. Encoder stack
		for enc_block in self.encoder:
			x_enc = enc_block(x_enc)
   
		# 3. Decoder stack
		for dec_block in self.decoder:
			x_dec = dec_block(x_dec, enc_kv_input=x_enc)
		
		linear_out = self.linear(x_dec)

		return linear_out