import argparse
import os
import torch
import torch.nn as nn
import copy
import math
import random

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

PAD, BOS, EOS = 0, 1, 2
CHARS = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ"
stoi = {ch: i + 3 for i, ch in enumerate(CHARS)}   # 'a' → 3, ..., 'z' → 28, ' ' → 29, 'A' → 30, ..., 'Z' → 55
itos = {i: ch for ch, i in stoi.items()}
VOCAB_SIZE = len(CHARS) + 3                          # 56
MAX_LEN = 50                                         # 학습 문자열 최대 길이

def make_batch(batch_size, min_len=1, max_len=MAX_LEN):
  srcs, tgts = [], []
  for _ in range(batch_size):
    s = "".join(random.choices(CHARS, k=random.randint(min_len, max_len)))
    srcs.append([stoi[c] for c in s] + [EOS])
    tgts.append([BOS] + [stoi[c] for c in reversed(s)] + [EOS])
  src = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in srcs], batch_first=True, padding_value=PAD)
  tgt = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in tgts], batch_first=True, padding_value=PAD)
  return src, tgt[:, :-1], tgt[:, 1:]   # src, tgt_in, tgt_out

def encode_string(s):
  return [stoi[c] for c in s] + [EOS]

def decode_tokens(tokens):
  chars = []
  for t in tokens:
    if t == EOS:
      break
    chars.append(itos.get(t, "?"))
  return "".join(chars)

# Learning rate schedule (5.3절, 식 3)
# lrate = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
# warmup_steps 까지 선형 증가 후 step^-0.5 에 비례하여 감소
def rate(step, embedding_dim, warmup_steps):
  step = max(step, 1)  # step=0 일 때 0^-0.5 방지
  return embedding_dim ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)

# 한 번에 한 토큰씩 생성: 매 step마다 지금까지 생성한 토큰 전체를 디코더에 다시 넣음
# src: (batch_size, src_length) -> (batch_size, <= max_length) 생성된 토큰 (BOS 제외)
@torch.no_grad()
def greedy_decode(model, src, max_length):
  src_mask = make_pad_mask(src, PAD)
  memory = model.encode(src, src_mask)  # 소스는 한 번만 인코딩

  batch_size = src.size(0)
  ys = torch.full((batch_size, 1), BOS, dtype=torch.long, device=src.device)
  finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
  for _ in range(max_length):
    tgt_mask = subsequent_mask(ys.size(1)).to(src.device)
    out = model.decode(memory, src_mask, ys, tgt_mask)
    logits = model.generator(out[:, -1])  # 마지막 위치만 사용: (batch_size, vocab_size)
    next_token = logits.argmax(dim=-1)
    next_token = next_token.masked_fill(finished, PAD)  # 이미 끝난 문장은 PAD로 채움
    ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
    finished |= next_token == EOS
    if finished.all():
      break
  return ys[:, 1:]

# 문자열 단위 완전 일치 정확도
def evaluate(model, val_batches, device):
  model.eval()
  correct, total = 0, 0
  for src, _, tgt_out in val_batches:
    src = src.to(device)
    pred = greedy_decode(model, src, max_length=src.size(1) + 1)
    for p, t in zip(pred.tolist(), tgt_out.tolist()):
      correct += decode_tokens(p) == decode_tokens(t)
      total += 1
  model.train()
  return correct / total

def parse_args():
  parser = argparse.ArgumentParser(description = "Transformer string reversal trainer")
  parser.add_argument(
    "--steps",
    type = int,
    default = 10000,
    help = "Number of training steps.",
  )
  parser.add_argument(
    "--batch_size",
    type = int,
    default = 128,
    help = "Batch size for training.",
  )
  parser.add_argument(
    "--warmup_steps",
    type = int,
    default = 400,
    help = "Warmup steps for the learning rate schedule (paper: 4000).",
  )
  parser.add_argument(
    "--device",
    choices = ["cpu", "cuda", "mps"],
    default = "mps",
    help = "Choose whether to run on CPU, CUDA, or MPS.",
  )
  parser.add_argument(
    "--save_path",
    default = "./saved_models/reverse_model.pth",
    help = "Path to save the trained model (or to load it with --infer_mode).",
  )
  parser.add_argument(
    "--infer_mode",
    action = "store_true",
    help = "Load the saved model and reverse strings typed in the terminal.",
  )
  return parser.parse_args()

def get_device(device_name):
  if device_name == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested, but no CUDA device is available.")
  if device_name == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("MPS was requested, but no MPS device is available.")
  return torch.device(device_name)

EMBEDDING_DIM = 128

# 학습과 추론에서 같은 구조를 써야 state_dict를 불러올 수 있으므로 한 곳에서 생성
def build_model(device):
  model = make_model(VOCAB_SIZE, VOCAB_SIZE, Nx=2, embedding_dim=EMBEDDING_DIM, ff_dim=512, head_count=4,
                     share_src_tgt_embed=True)
  return model.to(device)

def reverse_string(model, s, device):
  src = torch.tensor([encode_string(s)], device=device)
  return decode_tokens(greedy_decode(model, src, max_length=len(s) + 5)[0].tolist())

def train(args, device):
  # 고정된 검증셋 (학습 데이터는 매 step 새로 생성)
  random.seed(42)
  val_batches = [make_batch(128) for _ in range(4)]
  random.seed()

  embedding_dim = EMBEDDING_DIM
  model = build_model(device)
  print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

  # Optimizer (5.3절): base lr=1.0 에 LambdaLR 이 rate()를 곱하므로 실제 lr = rate(step)
  optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
  scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda step: rate(step, embedding_dim, args.warmup_steps)
  )
  # Label Smoothing (5.4절), PAD 위치는 loss에서 제외
  criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=PAD)

  model.train()
  for step in range(1, args.steps + 1):
    src, tgt_in, tgt_out = (t.to(device) for t in make_batch(args.batch_size))
    src_mask = make_pad_mask(src, PAD)
    tgt_mask = make_pad_mask(tgt_in, PAD) & subsequent_mask(tgt_in.size(1)).to(device)

    logits = model(src, tgt_in, src_mask, tgt_mask)  # (batch_size, tgt_length, vocab_size)
    loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 100 == 0:
      print(f"step {step:5d} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.6f}")
    if step % 500 == 0 or step == args.steps:
      print(f"step {step:5d} | val exact match {evaluate(model, val_batches, device):.2%}")

  os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok = True)
  torch.save(model.state_dict(), args.save_path)
  print(f"Saved model to {args.save_path}")

  # 예시 출력 (학습 범위 밖의 길이 포함)
  model.eval()
  for s in ["hello", "Hello World", "Attention Is All You Need", CHARS[:26] + CHARS[27:]]:
    pred = reverse_string(model, s, device)
    mark = "O" if pred == s[::-1] else "X"
    print(f"[{mark}] {s!r} -> {pred!r}")

def infer(args, device):
  model = build_model(device)
  model.load_state_dict(torch.load(args.save_path, map_location=device))
  model.eval()
  print(f"Loaded model from {args.save_path}")
  print("Type a string (a-z, A-Z, space) to reverse. Empty input or Ctrl+D to quit.")

  while True:
    try:
      # 공백도 입력 문자이므로 strip 하지 않음 (input()은 줄바꿈만 제거)
      s = input("> ")
    except (EOFError, KeyboardInterrupt):
      print()
      break
    if not s:
      break
    invalid = sorted(set(c for c in s if c not in stoi))
    if invalid:
      print(f"Unsupported characters: {''.join(invalid)!r} (only a-z, A-Z, space)")
      continue

    pred = reverse_string(model, s, device)
    mark = "O" if pred == s[::-1] else "X"
    note = f" (longer than training max {MAX_LEN})" if len(s) > MAX_LEN else ""
    # repr로 출력하여 앞뒤 공백이 보이도록 함
    print(f"[{mark}] {pred!r}{note}")

def main():
  args = parse_args()
  device = get_device(args.device)
  if args.infer_mode:
    infer(args, device)
  else:
    train(args, device)

if __name__ == "__main__":
  main()
