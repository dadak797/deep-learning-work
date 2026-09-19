import torch
import torch.nn as nn
import copy
import math

# Encoder
# ├─ EncoderLayer 1
# │  ├─ SelfAttention → Dropout → Add → LayerNorm
# │  └─ FeedForward   → Dropout → Add → LayerNorm
# ├─ EncoderLayer 2
# │  └─ ...
# ├─ ...
# └─ EncoderLayer 6
#    └─ ...

def scaled_dot_product_attention(query, key, value, mask=None):
  # query, key, value: (batch_size, head_count, seq_length, key_dim or value_dim)
  key_dim = query.size(-1)
  # score: (batch_size, head_count, seq_length, seq_length)
  scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key_dim)
  if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
  attn = torch.softmax(scores, dim=-1)
  return torch.matmul(attn, value), attn

class MultiHeadAttention(nn.Module):
  def __init__(self, head_count, embedding_dim):
    super().__init__()
    self.head_count = head_count
    self.embedding_dim = embedding_dim
    assert embedding_dim % head_count == 0, "embedding_dim must be divisible by head_count"
    self.key_dim = embedding_dim // head_count
    self.value_dim = self.key_dim

    # 하나로 정의한 후에 각 헤드로 분리하여 사용함
    # Query Matrix: (embedding_dim, head_count * key_dim)
    # Key Matrix: (embedding_dim, head_count * key_dim)
    # Value Matrix: (embedding_dim, head_count * value_dim)
    self.query_mat = nn.Linear(embedding_dim, head_count * self.key_dim)
    self.key_mat = nn.Linear(embedding_dim, head_count * self.key_dim)
    self.value_mat = nn.Linear(embedding_dim, head_count * self.value_dim)

    # Output Matrix: (embedding_dim, embedding_dim)
    self.output_mat = nn.Linear(embedding_dim, embedding_dim)

    self.attn = None

  # query, key, value: (batch_size, seq_length, embedding_dim)
  def forward(self, query, key, value, mask=None):
    if mask is not None:
      # Input Mask: (batch_size, seq_length, seq_length)
      # Expand mask for multi-head attention: (batch_size, 1, seq_length, seq_length)
      mask = mask.unsqueeze(1)

    batch_size = query.size(0)
    query_length = query.size(1)
    key_length = key.size(1)
    value_length = value.size(1)
    assert key_length == value_length, "Key and Value sequence lengths must match"

    # Linear projections
    query = self.query_mat(query)  # (batch_size, seq_length, head_count * key_dim)
    key = self.key_mat(key)        # (batch_size, seq_length, head_count * key_dim)
    value = self.value_mat(value)  # (batch_size, seq_length, head_count * value_dim)

    # Reshape for multi-head attention
    # view: (batch_size, seq_length, head_count * key_dim) -> (batch_size, seq_length, head_count, key_dim)
    # transpose(1, 2): (batch_size, head_count, seq_length, key_dim or value_dim)
    query = query.view(batch_size, query_length, self.head_count, self.key_dim).transpose(1, 2)
    key = key.view(batch_size, key_length, self.head_count, self.key_dim).transpose(1, 2)
    value = value.view(batch_size, value_length, self.head_count, self.value_dim).transpose(1, 2)

    # Apply scaled dot-product attention
    x, attn = scaled_dot_product_attention(query, key, value, mask)
    self.attn = attn.detach()

    # Concatenate heads and put through final linear layer
    # transpose(1, 2): (batch_size, head_count, seq_length, value_dim) -> (batch_size, seq_length, head_count, value_dim)
    # contiguous: 메모리 상에서 연속된 형태로 변환하여 view 연산이 안전하게 수행되도록 함
    # view: (batch_size, seq_length, head_count, value_dim) -> (batch_size, seq_length, head_count * value_dim)
    x = x.transpose(1, 2).contiguous().view(batch_size, query_length, self.head_count * self.value_dim)
    
    return self.output_mat(x)

class EncoderLayer(nn.Module):
  def __init__(self, embedding_dim, self_attn, feed_forward, dropout=0.1, epsilon=1e-6):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.epsilon = epsilon

    # Self-Attention SubLayer
    self.self_attn = self_attn
    self.dropout1 = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(embedding_dim, eps=epsilon)

    # Feed-Forward SubLayer
    self.feed_forward = feed_forward
    self.dropout2 = nn.Dropout(dropout)
    self.norm2 = nn.LayerNorm(embedding_dim, eps=epsilon)

  # x: (batch_size, seq_length, embedding_dim)
  # LayerNorm: 정수를 입력으로 받으면 마지막 차원(embedding_dim)의 차원에 대해 Normalization이 적용됨
  def forward(self, x, mask):
    attn_output = self.self_attn(x, x, x, mask)  # Self-Attention
    x = x + self.dropout1(attn_output)           # Add
    x = self.norm1(x)                            # LayerNorm1
    ff_output = self.feed_forward(x)             # Feed-Forward
    x = x + self.dropout2(ff_output)             # Add
    x = self.norm2(x)                            # LayerNorm2
    return x

class Encoder(nn.Module):
  def __init__(self, layer, Nx):
    super().__init__()
    self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(Nx)])

    # Initialize parameters using Xavier uniform initialization
    for parameter in self.parameters():
      if parameter.dim() > 1:
        nn.init.xavier_uniform_(parameter)

  def forward(self, x, mask):
    for layer in self.layers:
      x = layer(x, mask)
    return x

if __name__ == "__main__":
  pass  
