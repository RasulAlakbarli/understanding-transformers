""" Code inspired from a video by Andrej Karpathy. Link: https://www.youtube.com/watch?v=zduSFxRajkE """

from collections import defaultdict
from math import inf
import tqdm

class BPE_Tokenizer:
	"""
	Byte Pair Encoding (BPE) tokenizer implementation.
	
	Args:	
		text (str): The input text to train the tokenizer on.
	"""
	def __init__(self, text: str):
		self.text = text
		self.text_tokens = list(text.encode(encoding="utf-8"))
		self.merges = defaultdict(int)
  
	def get_max_pair(self, tokens: list):
		"""
		Finds the most frequent adjacent token pair in the token list.
  
		Args:
			tokens (list): List of tokens to analyze.

		Returns:
			tuple: The most frequent adjacent token pair.
		"""
		pairs = defaultdict(int)
  
		for i in range(len(tokens)-1):
			pair = (tokens[i], tokens[i+1])
			pairs[pair] += 1
		if not pairs:
			return None

		max_pair = max(pairs, key=pairs.get)
		return max_pair

	def update_tokens(self, tokens: list, max_pair: tuple, new_id: int):
		"""
		Updates the token list by merging the specified pair into a new token.
		Args:
			tokens (list): List of tokens to update.
			max_pair (tuple): The token pair to merge.
			new_id (int): The new token id to replace the merged pair.
		Returns:
			list: The updated list of tokens.
    	"""
		new_vocab = []
		i = 0
		while i<len(tokens):
			if i == len(tokens)-1:
				new_vocab.append(tokens[i])
				i+=1
			else:
				pair = (tokens[i], tokens[i+1])
				if pair == max_pair:
					new_vocab.append(new_id)
					i+=2
				else:
					new_vocab.append(tokens[i])
					i+=1   
		return new_vocab
        
	def fit(self, n_repeats: int):
		"""
		Trains the BPE tokenizer by performing the specified number of merge operations.
		Args:
			n_repeats (int): Number of merge operations to perform.
		"""
		for i in tqdm.tqdm(range(n_repeats), desc="Training BPE Tokenizer"):
			new_id = 256 + i # max id of a character from utf-8 is 255
			max_pair = self.get_max_pair(self.text_tokens)
			if not max_pair:
				break
			self.merges[max_pair] = new_id
			self.text_tokens = self.update_tokens(self.text_tokens, max_pair, new_id)
		print("Total number of tokens:", len(set(self.text_tokens)))
   
	def text2tok(self, text: str):
		""" 
		Converts input text to a list of token ids using the learned merges.
  
		Args:
			text (str): The input text to tokenize.

		Returns:
			list: The list of token ids.
		"""
		# Start from raw UTF-8 bytes
		tokens = list(text.encode("utf-8"))
		
		# Apply all learned merges in training order
		for pair, idx in self.merges.items():
			tokens = self.update_tokens(tokens, pair, idx)
		
		return tokens
		
	def tok2text(self, ids: list):
		""" 
		Converts a list of token ids back to the original text using the learned merges.

		Args:
			ids (list): The list of token ids to convert.

		Returns:
			str: The original text.
		"""
		vocab = {idx: bytes([idx]) for idx in range(256)}
		for (p0, p1), idx in self.merges.items():
			vocab[idx] = vocab[p0] + vocab[p1]
		tokens = b"".join(vocab[idx] for idx in ids)
		text = tokens.decode("utf-8", errors="replace")
		return text