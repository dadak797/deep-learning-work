# 학습된 모델과 학습 로그로 그래프를 저장 (main.py는 수정하지 않고 함수만 가져다 씀)
#
#   mkdir -p logs/reverse
#   python -u main.py --task reverse | tee logs/reverse/train.log   # 학습 + 출력 로그 저장
#   python analyze.py --task reverse                                # 그래프 저장
#
# main.py는 if __name__ == "__main__": 아래에서만 학습을 시작하므로 import해도 학습이 실행되지 않음
import argparse
import math
import os
import random
import re
import torch
import torch.nn.functional as F
import plots
from main import (PAD, BOS, VOCAB_SIZE, MAX_LEN, CHARS, EMBEDDING_DIM, TASKS, stoi, make_batch, encode_string,
                  decode_tokens, greedy_decode, make_pad_mask, subsequent_mask, rate, build_model, get_device)

LABEL_SMOOTHING = 0.1              # main.py의 CrossEntropyLoss(label_smoothing=0.1)과 같은 값
LENGTH_EVAL_MAX = MAX_LEN + 15     # 길이별 정확도: 학습 범위 밖 길이까지 확인
LENGTH_EVAL_SAMPLES = 32           # 길이마다 평가할 문자열 수
ATTENTION_EXAMPLE = "Attention Is All You Need"
TOKEN_LABELS = ["<pad>", "<s>", "</s>"] + ["␣" if c == " " else c for c in CHARS]

# main.py의 출력 형식:
#   step   100 | loss 3.7986 | lr 0.001105
#   step   500 | val exact match 29.10%
LOSS_LINE = re.compile(r"step\s+(\d+) \| loss ([\d.]+) \| lr ([\d.]+)")
VAL_LINE = re.compile(r"step\s+(\d+) \| val exact match ([\d.]+)%")

def parse_log(path):
  history = {"log_step": [], "loss": [], "lr": [], "eval_step": [], "val_exact_match": []}
  with open(path) as f:
    for line in f:
      if match := LOSS_LINE.search(line):
        history["log_step"].append(int(match[1]))
        history["loss"].append(float(match[2]))
        history["lr"].append(float(match[3]))
      elif match := VAL_LINE.search(line):
        history["eval_step"].append(int(match[1]))
        history["val_exact_match"].append(float(match[2]) / 100)
  return history if history["log_step"] else None

# label smoothing의 정답 분포: 정답 1 - ε + ε/C, 나머지 ε/C (PyTorch 방식)
# 모델이 이 분포를 그대로 맞혀도 cross-entropy는 이 분포의 엔트로피만큼 남음 (loss 하한)
def label_smoothing_stats(epsilon, class_count):
  target = 1 - epsilon + epsilon / class_count
  other = epsilon / class_count
  floor = -(target * math.log(target) + (class_count - 1) * other * math.log(other))
  return target, floor

# main.py의 train()과 같은 방법으로 검증셋을 다시 만듦 (seed 42)
def make_val_batches(task):
  random.seed(42)
  val_batches = [make_batch(task, 128) for _ in range(4)]
  random.seed()
  return val_batches

# 생성 결과를 정답과 같은 길이로 맞춤 (짧으면 PAD로 채우고 길면 자름)
def align_length(pred, length):
  if pred.size(1) < length:
    return F.pad(pred, (0, length - pred.size(1)), value=PAD)
  return pred[:, :length]

# 정확히 length 글자인 문자열만 모음 (word_reverse는 끝 공백 제거로 짧아질 수 있음)
def sample_exact_length(generate, length, count):
  strings = []
  while len(strings) < count:
    s = generate(length, length)
    if len(s) == length:
      strings.append(s)
  return strings

# 학습된 모델을 분석하여 그래프용 데이터를 모음
@torch.no_grad()
def collect_analysis(model, task, device):
  model.eval()
  generate, target = TASKS[task]["generate"], TASKS[task]["target"]
  analysis = {"lengths": [], "length_exact_match": [], "length_token_acc": [], "position_error": [],
              "samples_per_length": LENGTH_EVAL_SAMPLES}

  # 1. 길이별 정확도와 출력 위치별 오류율 (greedy)
  random.seed(0)
  for length in range(1, LENGTH_EVAL_MAX + 1):
    strings = sample_exact_length(generate, length, LENGTH_EVAL_SAMPLES)
    src = torch.tensor([encode_string(s) for s in strings], device=device)
    tgt_out = torch.tensor([encode_string(target(s)) for s in strings], device=device)
    pred = align_length(greedy_decode(model, src, max_length=length + 2), length + 1)
    correct = (pred == tgt_out).float()  # (samples, length + 1)
    analysis["lengths"].append(length)
    analysis["length_exact_match"].append(
      sum(decode_tokens(p) == target(s) for p, s in zip(pred.tolist(), strings)) / len(strings))
    analysis["length_token_acc"].append(correct.mean().item())
    analysis["position_error"].append((1 - correct.mean(dim=0)).tolist())

  # 2. 검증셋 teacher forcing 예측: 문자 혼동 행렬, 예측 확신도
  confusion = torch.zeros(VOCAB_SIZE, VOCAB_SIZE, dtype=torch.long)
  max_probs, corrects = [], []
  for src, tgt_in, tgt_out in make_val_batches(task):
    src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
    tgt_mask = make_pad_mask(tgt_in, PAD) & subsequent_mask(tgt_in.size(1)).to(device)
    logits = model(src, tgt_in, make_pad_mask(src, PAD), tgt_mask)
    max_prob, pred = logits.softmax(dim=-1).max(dim=-1)
    non_pad = tgt_out != PAD
    truth, pred = tgt_out[non_pad].cpu(), pred[non_pad].cpu()
    confusion.index_put_((truth, pred), torch.ones_like(truth), accumulate=True)
    max_probs.append(max_prob[non_pad].cpu())
    corrects.append(truth == pred)
  analysis["confusion"] = confusion.tolist()
  analysis["max_prob"] = torch.cat(max_probs).tolist()
  analysis["correct"] = torch.cat(corrects).tolist()

  # 3. 예시 문장 하나의 attention 가중치 (MultiHeadAttention이 마지막 forward의 attn을 저장해 둠)
  src = torch.tensor([encode_string(ATTENTION_EXAMPLE)], device=device)
  tgt_in = torch.tensor([[BOS] + [stoi[c] for c in target(ATTENTION_EXAMPLE)]], device=device)
  model(src, tgt_in, make_pad_mask(src, PAD), subsequent_mask(tgt_in.size(1)).to(device))
  analysis["attention"] = {
    "example": ATTENTION_EXAMPLE,
    "src_labels": [TOKEN_LABELS[t] for t in src[0].tolist()],
    "tgt_labels": [TOKEN_LABELS[t] for t in tgt_in[0].tolist()],
    "encoder_self": [layer.self_attn.attn[0].cpu().numpy() for layer in model.encoder.layers],
    "decoder_self": [layer.self_attn.attn[0].cpu().numpy() for layer in model.decoder.layers],
    "cross": [layer.src_attn.attn[0].cpu().numpy() for layer in model.decoder.layers],
  }

  # 4. 고정값/가중치: positional encoding, (공유된) 토큰 임베딩
  analysis["positional_encoding"] = model.src_embed[1].pe[0, :LENGTH_EVAL_MAX + 1].cpu().numpy()
  analysis["embedding"] = model.src_embed[0].embedding.weight.detach().cpu().numpy()
  return analysis

def parse_args():
  parser = argparse.ArgumentParser(description = "Save plots from a trained model and its training log")
  parser.add_argument(
    "--task",
    choices = list(TASKS),
    default = "reverse",
    help = "Task of the trained model.",
  )
  parser.add_argument(
    "--save_path",
    default = None,
    help = "Path of the trained model. Default depends on --task (same as main.py).",
  )
  parser.add_argument(
    "--log_dir",
    default = None,
    help = "Folder that has train.log and receives the plots. Default: ./logs/<task>.",
  )
  parser.add_argument(
    "--warmup_steps",
    type = int,
    default = 400,
    help = "Warmup steps used for training (for the learning rate plot).",
  )
  parser.add_argument(
    "--device",
    choices = ["cpu", "cuda", "mps"],
    default = "mps",
    help = "Choose whether to run on CPU, CUDA, or MPS.",
  )
  args = parser.parse_args()
  if args.save_path is None:
    args.save_path = TASKS[args.task]["save_path"]
  if args.log_dir is None:
    args.log_dir = os.path.join("./logs", args.task)
  return args

def main():
  args = parse_args()
  device = get_device(args.device)
  model = build_model(device)
  model.load_state_dict(torch.load(args.save_path, map_location=device))
  print(f"Task: {args.task}")
  print(f"Loaded model from {args.save_path}")

  history = None
  log_path = os.path.join(args.log_dir, "train.log")
  if os.path.exists(log_path):
    history = parse_log(log_path)
  if history is None:
    print(f"No training log in {log_path}: skipping the loss and validation accuracy curves")

  # 학습률은 rate()로 계산 (로그가 있으면 로그의 마지막 step까지)
  last_step = history["log_step"][-1] if history is not None else 10000
  steps = list(range(1, last_step + 1))
  lr_schedule = (steps, [rate(step, EMBEDDING_DIM, args.warmup_steps) for step in steps])

  print("Analyzing the model...")
  analysis = collect_analysis(model, args.task, device)
  smoothing_target, loss_floor = label_smoothing_stats(LABEL_SMOOTHING, VOCAB_SIZE)
  paths = plots.save_all(args.log_dir, history, lr_schedule, analysis, TOKEN_LABELS, MAX_LEN,
                         args.warmup_steps, loss_floor, smoothing_target)
  print(f"Saved {len(paths)} plots to {args.log_dir}")

if __name__ == "__main__":
  main()
