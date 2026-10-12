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
#
# DecoderLayer
# ├─ Masked SelfAttention        → Dropout → Add → LayerNorm
# ├─ Encoder-Decoder Attention   → Dropout → Add → LayerNorm
# └─ FeedForward                 → Dropout → Add → LayerNorm

def scaled_dot_product_attention(query, key, value, mask=None):
  # query, key, value: (batch_size, head_count, seq_length, key_dim or value_dim)
  key_dim = query.size(-1)
  # score: (batch_size, head_count, seq_length, seq_length)
  scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key_dim)
  if mask is not None:
    scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
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
      # Input Mask (1 = attend, 0 = masked):
      #   - padding mask:    (batch_size, 1, key_length)
      #   - look-ahead mask: (batch_size, query_length, key_length)
      # Expand mask for multi-head attention (broadcast over heads):
      #   -> (batch_size, 1, 1 or query_length, key_length)
      assert mask.dim() == 3, "mask must be 3D: (batch_size, 1 or query_length, key_length)"
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

class PositionwiseFeedForward(nn.Module):
  def __init__(self, embedding_dim, ff_dim=2048):
    super().__init__()
    # W1: (embedding_dim, ff_dim), W2: (ff_dim, embedding_dim)
    self.linear1 = nn.Linear(embedding_dim, ff_dim)
    self.linear2 = nn.Linear(ff_dim, embedding_dim)

  # x: (batch_size, seq_length, embedding_dim)
  def forward(self, x):
    x = torch.relu(self.linear1(x))  # (batch_size, seq_length, ff_dim)
    return self.linear2(x)           # (batch_size, seq_length, embedding_dim)

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

class DecoderLayer(nn.Module):
  def __init__(self, embedding_dim, self_attn, src_attn, feed_forward, dropout=0.1, epsilon=1e-6):
    super().__init__()
    # 같은 인스턴스를 넘기면 두 Attention이 가중치를 공유하게 되므로 막음
    assert self_attn is not src_attn, "self_attn and src_attn must be different instances"

    # Masked Self-Attention SubLayer
    self.self_attn = self_attn
    self.dropout1 = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(embedding_dim, eps=epsilon)

    # Encoder-Decoder Attention SubLayer
    self.src_attn = src_attn
    self.dropout2 = nn.Dropout(dropout)
    self.norm2 = nn.LayerNorm(embedding_dim, eps=epsilon)

    # Feed-Forward SubLayer
    self.feed_forward = feed_forward
    self.dropout3 = nn.Dropout(dropout)
    self.norm3 = nn.LayerNorm(embedding_dim, eps=epsilon)

  # x: (batch_size, tgt_length, embedding_dim) 디코더 입력
  # memory: (batch_size, src_length, embedding_dim) 인코더 출력
  # src_mask: (batch_size, 1, src_length) padding mask
  # tgt_mask: (batch_size, tgt_length, tgt_length) padding mask & subsequent mask
  def forward(self, x, memory, src_mask, tgt_mask):
    attn_output = self.self_attn(x, x, x, tgt_mask)              # Masked Self-Attention
    x = x + self.dropout1(attn_output)                           # Add
    x = self.norm1(x)                                            # LayerNorm1
    # Query: 디코더, Key/Value: 인코더 출력
    attn_output = self.src_attn(x, memory, memory, src_mask)     # Encoder-Decoder Attention
    x = x + self.dropout2(attn_output)                           # Add
    x = self.norm2(x)                                            # LayerNorm2
    ff_output = self.feed_forward(x)                             # Feed-Forward
    x = x + self.dropout3(ff_output)                             # Add
    x = self.norm3(x)                                            # LayerNorm3
    return x

class PositionalEncoding(nn.Module):
  def __init__(self, embedding_dim, dropout=0.1, max_length=5000):
    super().__init__()
    self.dropout = nn.Dropout(dropout)

    # pe: (max_length, embedding_dim)
    pe = torch.zeros(max_length, embedding_dim)
    # position: (max_length, 1) → pos
    position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
    # two_i: (embedding_dim / 2,) → 2i = [0, 2, 4, ..., embedding_dim - 2]
    two_i = torch.arange(0, embedding_dim, 2, dtype=torch.float)
    # denominator: 10000^(2i / embedding_dim)
    denominator = torch.pow(10000.0, two_i / embedding_dim)

    # broadcast: (max_length, 1) / (embedding_dim / 2,) -> (max_length, embedding_dim / 2)
    pe[:, 0::2] = torch.sin(position / denominator)  # PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    pe[:, 1::2] = torch.cos(position / denominator)  # PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    # (1, max_length, embedding_dim): batch 차원으로 broadcast 되도록 함
    # register_buffer: 학습 파라미터는 아니지만 state_dict에 저장되고 .to(device) 시 함께 이동함
    self.register_buffer("pe", pe.unsqueeze(0))

  # x: (batch_size, seq_length, embedding_dim)
  def forward(self, x):
    x = x + self.pe[:, :x.size(1)]
    return self.dropout(x)

class Embeddings(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.embedding_dim = embedding_dim

  # x: (batch_size, seq_length) 토큰 인덱스
  def forward(self, x):
    return self.embedding(x) * math.sqrt(self.embedding_dim)

# src: (batch_size, src_length) → (batch_size, 1, src_length)
def make_pad_mask(src, pad_idx):
  return (src != pad_idx).unsqueeze(1)

# Look-ahead mask: 위치 i는 j <= i 인 위치만 볼 수 있음 (1 = attend, 0 = masked)
# size=4 일 때:
#   [[1, 0, 0, 0],
#    [1, 1, 0, 0],
#    [1, 1, 1, 0],
#    [1, 1, 1, 1]]
# 반환: (1, size, size) → batch 차원으로 broadcast 되어 padding mask와 & 로 결합 가능
#   tgt_mask = make_pad_mask(tgt, pad_idx) & subsequent_mask(tgt.size(1)).to(tgt.device)
#   (batch_size, 1, tgt_length) & (1, tgt_length, tgt_length) -> (batch_size, tgt_length, tgt_length)
def subsequent_mask(size):
  return torch.tril(torch.ones(1, size, size, dtype=torch.bool))

class Encoder(nn.Module):
  def __init__(self, layer, Nx):
    super().__init__()
    self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(Nx)])

  def forward(self, x, mask):
    for layer in self.layers:
      x = layer(x, mask)
    return x

class Decoder(nn.Module):
  def __init__(self, layer, Nx):
    super().__init__()
    self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(Nx)])

  # x: (batch_size, tgt_length, embedding_dim)
  # memory: (batch_size, src_length, embedding_dim) 인코더 출력 (모든 층이 같은 memory를 참조함)
  def forward(self, x, memory, src_mask, tgt_mask):
    for layer in self.layers:
      x = layer(x, memory, src_mask, tgt_mask)
    return x

# 디코더 출력을 다음 토큰에 대한 점수(logits)로 변환 (Figure 1의 Linear)
# Softmax는 여기서 적용하지 않음:
#   - 학습: nn.CrossEntropyLoss가 내부에서 log_softmax를 적용함
#   - 추론: greedy decoding은 argmax만 필요하므로 softmax가 필요 없음
class Generator(nn.Module):
  def __init__(self, embedding_dim, vocab_size):
    super().__init__()
    self.proj = nn.Linear(embedding_dim, vocab_size)

  # x: (batch_size, tgt_length, embedding_dim) -> (batch_size, tgt_length, vocab_size)
  def forward(self, x):
    return self.proj(x)

class Transformer(nn.Module):
  def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed  # Embeddings → PositionalEncoding
    self.tgt_embed = tgt_embed  # Embeddings → PositionalEncoding
    self.generator = generator

  # src: (batch_size, src_length) 토큰 인덱스
  # src_mask: (batch_size, 1, src_length)
  # 반환 memory: (batch_size, src_length, embedding_dim)
  def encode(self, src, src_mask):
    return self.encoder(self.src_embed(src), src_mask)

  # tgt: (batch_size, tgt_length) 토큰 인덱스 (shifted right: <bos>로 시작)
  # tgt_mask: (batch_size, tgt_length, tgt_length)
  # 반환: (batch_size, tgt_length, embedding_dim) Generator 적용 전 디코더 출력
  def decode(self, memory, src_mask, tgt, tgt_mask):
    return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

  # 학습용: 전체 타깃 시퀀스를 한 번에 처리 (teacher forcing)
  # 반환 logits: (batch_size, tgt_length, tgt_vocab_size)
  def forward(self, src, tgt, src_mask, tgt_mask):
    memory = self.encode(src, src_mask)
    return self.generator(self.decode(memory, src_mask, tgt, tgt_mask))

# 기본값은 논문의 base 모델 (Table 3)
# share_tgt_embed_generator: 타깃 임베딩 ↔ Generator 가중치 공유 (3.4절)
# share_src_tgt_embed: 소스 ↔ 타깃 임베딩 가중치 공유 (소스/타깃이 같은 vocab을 쓸 때만 사용)
def make_model(src_vocab_size, tgt_vocab_size, Nx=6, embedding_dim=512, ff_dim=2048, head_count=8, dropout=0.1,
               share_tgt_embed_generator=True, share_src_tgt_embed=False):
  c = copy.deepcopy
  attn = MultiHeadAttention(head_count, embedding_dim)
  ff = PositionwiseFeedForward(embedding_dim, ff_dim)

  src_embed = Embeddings(src_vocab_size, embedding_dim)
  tgt_embed = Embeddings(tgt_vocab_size, embedding_dim)
  generator = Generator(embedding_dim, tgt_vocab_size)

  # 같은 nn.Parameter 객체를 가리키게 하여 공유함 (크기는 모두 (vocab_size, embedding_dim))
  if share_tgt_embed_generator:
    generator.proj.weight = tgt_embed.embedding.weight
  if share_src_tgt_embed:
    assert src_vocab_size == tgt_vocab_size, "shared embeddings require the same vocab"
    src_embed.embedding.weight = tgt_embed.embedding.weight

  model = Transformer(
    Encoder(EncoderLayer(embedding_dim, c(attn), c(ff), dropout), Nx),
    Decoder(DecoderLayer(embedding_dim, c(attn), c(attn), c(ff), dropout), Nx),
    nn.Sequential(src_embed, PositionalEncoding(embedding_dim, dropout)),
    nn.Sequential(tgt_embed, PositionalEncoding(embedding_dim, dropout)),
    generator,
  )

  # Initialize parameters using Xavier uniform initialization
  # 공유된 가중치는 parameters()에 한 번만 나타나므로 한 번만 초기화됨
  # bias, LayerNorm(1D)은 PyTorch 기본값(0, 1) 유지
  for parameter in model.parameters():
    if parameter.dim() > 1:
      nn.init.xavier_uniform_(parameter)

  return model

if __name__ == "__main__":
  pass  
