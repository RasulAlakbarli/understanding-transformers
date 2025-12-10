import math
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
	"""
	Scaled Dot-Product Attention
	"""
	def __init__(self, d_attn):
		super().__init__()
		self.d_attn = d_attn
  
		self.W_Q = nn.Linear(self.d_attn, self.d_attn)
		self.W_K = nn.Linear(self.d_attn, self.d_attn)
		self.W_V = nn.Linear(self.d_attn, self.d_attn)
		self.softmax = nn.Softmax(dim=-1)
	def forward(self, x, mask=None):
		Q = self.W_Q(x)
		K = self.W_K(x)
		V = self.W_V(x)
		attn = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(Q.size(-1))
		if mask is not None:
			attn = attn.masked_fill(mask == 0, float("-1e9"))
		scores = self.softmax(attn)
  
		return torch.matmul(scores, V)

class MultiHeadAttention(nn.Module):
	"""
	Multi Head Attention
	"""
	def __init__(self, d_model, n_heads):
		super().__init__()
		assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
		self.d_model = d_model
		self.n_heads = n_heads
		self.d_head = d_model // n_heads
  
		self.W_Q = nn.Linear(self.d_model, self.d_model)
		self.W_K = nn.Linear(self.d_model, self.d_model)
		self.W_V = nn.Linear(self.d_model, self.d_model)
		self.dropout = nn.Dropout(0.2)
		self.softmax = nn.Softmax(dim=-1)
		self.final_linear = nn.Linear(self.d_model, self.d_model)
  
	def forward(self, x, mask=None, kv_input=None):
		Q = self.W_Q(x)
		if kv_input is not None:
			K = self.W_K(kv_input)
			V = self.W_V(kv_input)
		else:
			K = self.W_K(x)
			V = self.W_V(x)
   
		batch_size, dec_seq_len, d_model = Q.size()
		enc_seq_len = K.size(1)

		Q = Q.view(batch_size, dec_seq_len, self.n_heads, self.d_head).transpose(1, 2) # (batch_size, n_heads, seq_len, d_head)
		K = K.view(batch_size, enc_seq_len, self.n_heads, self.d_head).transpose(1, 2) # (batch_size, n_heads, seq_len, d_head)
		V = V.view(batch_size, enc_seq_len, self.n_heads, self.d_head).transpose(1, 2) # (batch_size, n_heads, seq_len, d_head)

		attn = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(Q.size(-1)) # (batch_size, n_heads, seq_len, seq_len)
		if mask is not None:
			attn = attn.masked_fill(mask == 0, float("-1e9"))

		scores = self.softmax(attn)
		x = self.dropout(scores)
		x = torch.matmul(x, V) # (batch_size, n_heads, seq_len, d_head)
		x = x.transpose(1, 2).contiguous().view(batch_size, dec_seq_len, self.d_model)

		return self.final_linear(x)