# 학습 로그(history)와 학습된 모델의 분석 결과(analysis)를 PNG 그래프로 저장
# analyze.py가 데이터를 모아서 넘기고, 여기서는 그리기만 함
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter

# 색상
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SHADE = "#f0efec"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # 한 그래프에 최대 3개 계열
SEQUENTIAL = LinearSegmentedColormap.from_list(
  "sequential_blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEQUENTIAL.set_bad(SURFACE)  # NaN(값 없음) 칸은 배경색
DIVERGING = LinearSegmentedColormap.from_list(
  "diverging_blue_red", ["#0d366b", "#3987e5", SHADE, "#e66767", "#8f2424"])

plt.rcParams.update({
  "figure.facecolor": SURFACE,
  "axes.facecolor": SURFACE,
  "savefig.facecolor": SURFACE,
  "axes.edgecolor": AXIS,
  "axes.labelcolor": INK_SECONDARY,
  "axes.titlecolor": INK,
  "axes.titlesize": 12,
  "axes.spines.top": False,
  "axes.spines.right": False,
  "axes.grid": True,
  "grid.color": GRID,
  "grid.linewidth": 0.8,
  "xtick.color": MUTED,
  "ytick.color": MUTED,
  "xtick.labelcolor": INK_SECONDARY,
  "ytick.labelcolor": INK_SECONDARY,
  "text.color": INK,
  "font.size": 10,
  "lines.linewidth": 2,
  "legend.frameon": False,
  "savefig.dpi": 150,
  "savefig.bbox": "tight",
})

def _save(fig, out_dir, name):
  path = os.path.join(out_dir, name)
  fig.savefig(path)
  plt.close(fig)
  return path

# 로그 축 눈금을 1, 2, 3, 5 단위의 일반 숫자로 표시 (기본값은 4×10^0 형태)
def _plain_log_axis(ax):
  ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 3, 5)))
  ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
  ax.yaxis.set_minor_formatter(NullFormatter())

def _heatmap_ticks(ax, row_labels, col_labels, max_labels=60, fontsize=7):
  if len(col_labels) <= max_labels:
    # 라벨이 많으면 <pad> 같은 긴 라벨끼리 겹치므로 세로로 회전
    rotation = 90 if len(col_labels) > 30 else 0
    ax.set_xticks(range(len(col_labels)), col_labels, fontsize=fontsize, rotation=rotation)
  if len(row_labels) <= max_labels:
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=fontsize)
  ax.tick_params(length=0)
  ax.grid(False)

# ─── 학습 곡선 (history: main.py 출력 로그를 파싱한 값) ─────────────────────

def plot_loss(history, loss_floor, out_dir):
  fig, ax = plt.subplots(figsize=(9, 4.5))
  steps = history["log_step"]
  ax.plot(steps, history["loss"], color=SERIES[0], marker="o", markersize=3,
          label="Train loss (one batch, printed every 100 steps)")
  ax.axhline(loss_floor, color=MUTED, linestyle="--", linewidth=1)
  ax.annotate(f"label smoothing floor {loss_floor:.3f}", (steps[-1], loss_floor), xytext=(0, 4),
              textcoords="offset points", ha="right", va="bottom", color=INK_SECONDARY, fontsize=9)
  ax.set_yscale("log")
  _plain_log_axis(ax)
  ax.set_title("Training loss")
  ax.set_xlabel("Step")
  ax.set_ylabel("Cross-entropy (log scale)")
  ax.legend(loc="upper right")
  return _save(fig, out_dir, "01_loss.png")

# lr_schedule: rate()로 계산한 (steps, lr), history가 있으면 로그에 찍힌 값도 점으로 표시
def plot_learning_rate(lr_schedule, history, warmup_steps, out_dir):
  fig, ax = plt.subplots(figsize=(9, 4))
  steps, lr = lr_schedule
  ax.plot(steps, lr, color=SERIES[0], label="rate(step), Eq. 3")
  if history is not None:
    ax.plot(history["log_step"], history["lr"], color=SERIES[1], linestyle="none", marker="o",
            markersize=4, label="Printed in the training log")
  ax.axvline(warmup_steps, color=MUTED, linestyle="--", linewidth=1)
  peak = int(np.argmax(lr))
  ax.annotate(f"peak {lr[peak]:.5f} at step {steps[peak]}\n(warmup_steps = {warmup_steps})",
              (steps[peak], lr[peak]), xytext=(8, 6), textcoords="offset points",
              va="bottom", color=INK_SECONDARY, fontsize=9)
  ax.set_title("Learning rate schedule")
  ax.set_xlabel("Step")
  ax.set_ylabel("Learning rate")
  ax.set_ylim(0, lr[peak] * 1.3)
  ax.legend(loc="upper right")
  return _save(fig, out_dir, "02_learning_rate.png")

def plot_val_accuracy(history, out_dir):
  fig, ax = plt.subplots(figsize=(9, 4.5))
  ax.plot(history["eval_step"], np.array(history["val_exact_match"]) * 100, color=SERIES[0],
          marker="o", markersize=5)
  ax.set_title("Validation exact match (512 strings, greedy)")
  ax.set_xlabel("Step")
  ax.set_ylabel("Exact match (%)")
  ax.set_ylim(0, 101)
  return _save(fig, out_dir, "03_val_exact_match.png")

# ─── 학습된 모델 분석 (analysis) ────────────────────────────────────────

def _shade_untrained_lengths(ax, max_train_len, last_length, label=True):
  ax.axvspan(max_train_len + 0.5, last_length + 0.5, color=SHADE, zorder=0)
  if label:
    ax.text(max_train_len + 1, 0.5, "longer than\ntraining data", transform=ax.get_xaxis_transform(),
            color=INK_SECONDARY, fontsize=9, va="center")

def plot_length_accuracy(analysis, max_train_len, out_dir):
  fig, ax = plt.subplots(figsize=(9, 4.5))
  lengths = analysis["lengths"]
  _shade_untrained_lengths(ax, max_train_len, lengths[-1])
  ax.plot(lengths, np.array(analysis["length_token_acc"]) * 100, color=SERIES[0], marker="o",
          markersize=3, label="Token accuracy")
  ax.plot(lengths, np.array(analysis["length_exact_match"]) * 100, color=SERIES[1], marker="o",
          markersize=3, label="Exact match")
  ax.set_title(f"Accuracy by input length ({analysis['samples_per_length']} strings per length, greedy)")
  ax.set_xlabel("Input length (characters)")
  ax.set_ylabel("Accuracy (%)")
  ax.set_xlim(0.5, lengths[-1] + 0.5)
  ax.set_ylim(0, 101)
  ax.legend(loc="lower left")
  return _save(fig, out_dir, "04_accuracy_by_length.png")

def plot_position_errors(analysis, max_train_len, out_dir):
  lengths = analysis["lengths"]
  # 행: 입력 길이, 열: 출력 위치 (길이 + EOS), 해당 길이에 없는 위치는 NaN
  width = lengths[-1] + 1
  matrix = np.full((len(lengths), width), np.nan)
  for row, errors in enumerate(analysis["position_error"]):
    matrix[row, :len(errors)] = errors

  # 왼쪽: 전체 길이 (학습 범위 밖은 오류율이 커서 색 범위를 차지함)
  # 오른쪽: 학습 범위 안의 길이만, 색 범위를 그 안의 최댓값에 맞춰 작은 오류도 보이게 함
  trained_rows = sum(length <= max_train_len for length in lengths)
  panels = ((matrix, lengths[-1], "All lengths"),
            (matrix[:trained_rows, :max_train_len + 1], max_train_len, "Training lengths only (rescaled)"))

  fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
  for ax, (data, last_length, title) in zip(axes, panels):
    vmax = max(np.nanmax(data), 0.01)
    image = ax.imshow(data * 100, cmap=SEQUENTIAL, vmin=0, vmax=vmax * 100, origin="lower",
                      aspect="auto", interpolation="nearest",
                      extent=(-0.5, data.shape[1] - 0.5, lengths[0] - 0.5, last_length + 0.5))
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Error rate (%)", shrink=0.8)
    ax.set_title(title)
    ax.set_xlabel("Output position (0 = first character, last = </s>)")
    ax.set_ylabel("Input length (characters)")
  axes[0].axhline(max_train_len + 0.5, color=MUTED, linestyle="--", linewidth=1)
  axes[0].text(1, max_train_len, "↑ longer than training data", color=INK_SECONDARY, fontsize=9, va="top")
  fig.suptitle("Where errors happen: error rate by output position (greedy)", color=INK, fontsize=12)
  return _save(fig, out_dir, "05_error_by_position.png")

def plot_confusion(analysis, token_labels, out_dir):
  # PAD/BOS는 정답에 나오지 않으므로 EOS(2)부터 표시
  confusion = np.array(analysis["confusion"], dtype=float)[2:, 2:]
  labels = token_labels[2:]
  errors = confusion.copy()
  np.fill_diagonal(errors, 0)
  total_errors = int(errors.sum())
  errors[errors == 0] = np.nan

  fig, ax = plt.subplots(figsize=(10, 9))
  if total_errors == 0:
    ax.text(0.5, 0.5, "No errors on the validation set", ha="center", va="center",
            transform=ax.transAxes, color=INK_SECONDARY, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
  else:
    image = ax.imshow(errors, cmap=SEQUENTIAL, vmin=0, interpolation="nearest")
    _heatmap_ticks(ax, labels, labels, fontsize=6)
    fig.colorbar(image, ax=ax, label="Count", shrink=0.8)
  ax.set_title(f"Character confusions, errors only (teacher forcing, {total_errors} errors / "
               f"{int(confusion.sum())} tokens)")
  ax.set_xlabel("Predicted")
  ax.set_ylabel("Correct")
  return _save(fig, out_dir, "06_confusion_matrix.png")

def plot_confidence(analysis, smoothing_target, out_dir):
  max_prob = np.asarray(analysis["max_prob"])
  correct = np.asarray(analysis["correct"], dtype=bool)
  bins = np.linspace(0, 1, 51)

  fig, ax = plt.subplots(figsize=(9, 4.5))
  for color, mask, name in ((SERIES[0], correct, "Correct"), (SERIES[1], ~correct, "Wrong")):
    if mask.any():
      ax.hist(max_prob[mask], bins=bins, density=True, histtype="step", linewidth=2, color=color,
              label=f"{name} ({mask.sum():,} tokens)")
  ax.axvline(smoothing_target, color=MUTED, linestyle="--", linewidth=1)
  ax.annotate(f"label smoothing target {smoothing_target:.3f}", (smoothing_target, 1),
              xycoords=("data", "axes fraction"), xytext=(-4, -4), textcoords="offset points",
              ha="right", va="top", color=INK_SECONDARY, fontsize=9)
  ax.set_yscale("log")
  ax.set_title("Prediction confidence (max softmax probability, teacher forcing)")
  ax.set_xlabel("Max probability")
  ax.set_ylabel("Density (log scale)")
  ax.legend(loc="upper left")
  return _save(fig, out_dir, "07_confidence.png")

# maps: 층별 (head_count, rows, cols) 배열의 리스트
def plot_attention(maps, row_labels, col_labels, title, row_name, col_name, out_dir, name):
  layer_count, head_count = len(maps), maps[0].shape[0]
  fig, axes = plt.subplots(layer_count, head_count, figsize=(3.2 * head_count, 3.2 * layer_count + 0.6),
                           squeeze=False, constrained_layout=True)
  for layer in range(layer_count):
    for head in range(head_count):
      ax = axes[layer][head]
      image = ax.imshow(maps[layer][head], cmap=SEQUENTIAL, vmin=0, vmax=1, interpolation="nearest")
      _heatmap_ticks(ax, row_labels, col_labels, max_labels=40, fontsize=5)
      ax.set_title(f"Layer {layer + 1} · Head {head + 1}", fontsize=9)
      if head == 0:
        ax.set_ylabel(f"{row_name} (query)", fontsize=8)
      if layer == layer_count - 1:
        ax.set_xlabel(f"{col_name} (key)", fontsize=8)
  fig.colorbar(image, ax=axes, label="Attention weight", shrink=0.6)
  fig.suptitle(title, color=INK, fontsize=12)
  return _save(fig, out_dir, name)

def plot_positional_encoding(analysis, out_dir):
  pe = np.asarray(analysis["positional_encoding"])  # (positions, embedding_dim)
  normalized = pe / np.linalg.norm(pe, axis=1, keepdims=True)
  similarity = normalized @ normalized.T

  fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
  image = left.imshow(pe.T, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto", origin="lower",
                      interpolation="nearest")
  left.grid(False)
  left.set_title("Positional encoding values")
  left.set_xlabel("Position")
  left.set_ylabel("Dimension (even: sin, odd: cos)")
  fig.colorbar(image, ax=left, shrink=0.8)

  # 유사도가 모두 양수 근처라서 단색 계열로 실제 범위만 표시
  image = right.imshow(similarity, cmap=SEQUENTIAL, vmin=similarity.min(), vmax=1, origin="lower",
                       interpolation="nearest")
  right.grid(False)
  right.set_title("Cosine similarity between positions")
  right.set_xlabel("Position")
  right.set_ylabel("Position")
  fig.colorbar(image, ax=right, shrink=0.8)
  return _save(fig, out_dir, "11_positional_encoding.png")

def _token_groups(token_labels):
  groups = []
  for label in token_labels:
    if len(label) == 1 and label.islower():
      groups.append(0)
    elif len(label) == 1 and label.isupper():
      groups.append(1)
    else:
      groups.append(2)
  return np.array(groups)

def plot_embedding_pca(analysis, token_labels, out_dir):
  weight = np.asarray(analysis["embedding"])
  centered = weight - weight.mean(axis=0)
  _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
  points = centered @ components[:2].T
  explained = singular_values ** 2 / (singular_values ** 2).sum()
  groups = _token_groups(token_labels)

  fig, ax = plt.subplots(figsize=(9, 8))
  for group, (color, name) in enumerate(zip(SERIES, ("Lowercase", "Uppercase", "Space / special"))):
    mask = groups == group
    ax.scatter(points[mask, 0], points[mask, 1], s=50, color=color, edgecolors=SURFACE, linewidths=1.5,
               label=name, zorder=3)
  for (x, y), label in zip(points, token_labels):
    ax.annotate(label, (x, y), xytext=(5, 3), textcoords="offset points", fontsize=8, color=INK_SECONDARY)
  ax.set_title("Token embeddings (shared weight), PCA")
  ax.set_xlabel(f"PC1 ({explained[0]:.1%} of variance)")
  ax.set_ylabel(f"PC2 ({explained[1]:.1%} of variance)")
  ax.legend(loc="best")
  return _save(fig, out_dir, "12_embedding_pca.png")

def plot_embedding_similarity(analysis, token_labels, out_dir):
  # 임베딩들이 공통 방향을 공유해서 그대로 재면 모두 0.5 근처가 됨
  # 평균 벡터를 빼고(mean-centered) 재야 토큰 사이의 차이가 드러남
  weight = np.asarray(analysis["embedding"])
  centered = weight - weight.mean(axis=0)
  normalized = centered / np.linalg.norm(centered, axis=1, keepdims=True)
  similarity = normalized @ normalized.T
  limit = np.abs(similarity[~np.eye(len(similarity), dtype=bool)]).max()  # 대각선(=1) 제외

  fig, ax = plt.subplots(figsize=(10, 9))
  image = ax.imshow(similarity, cmap=DIVERGING, vmin=-limit, vmax=limit, interpolation="nearest")
  _heatmap_ticks(ax, token_labels, token_labels, fontsize=6)
  fig.colorbar(image, ax=ax, label="Cosine similarity (diagonal = 1, clipped)", shrink=0.8)
  ax.set_title("Cosine similarity between token embeddings (mean-centered)")
  return _save(fig, out_dir, "13_embedding_similarity.png")

# ─── 전체 저장 ─────────────────────────────────────────────────────────

# history가 None이면 (학습 로그 없음) loss와 검증 정확도 곡선은 건너뜀
def save_all(out_dir, history, lr_schedule, analysis, token_labels, max_train_len, warmup_steps, loss_floor,
             smoothing_target):
  os.makedirs(out_dir, exist_ok=True)
  paths = []
  if history is not None:
    paths.append(plot_loss(history, loss_floor, out_dir))
  paths.append(plot_learning_rate(lr_schedule, history, warmup_steps, out_dir))
  if history is not None and history["eval_step"]:
    paths.append(plot_val_accuracy(history, out_dir))

  attention = analysis["attention"]
  src_labels, tgt_labels = attention["src_labels"], attention["tgt_labels"]
  example = attention["example"]
  paths += [
    plot_length_accuracy(analysis, max_train_len, out_dir),
    plot_position_errors(analysis, max_train_len, out_dir),
    plot_confusion(analysis, token_labels, out_dir),
    plot_confidence(analysis, smoothing_target, out_dir),
    plot_attention(attention["encoder_self"], src_labels, src_labels,
                   f"Encoder self-attention: {example!r}", "Source", "Source",
                   out_dir, "08_attention_encoder_self.png"),
    plot_attention(attention["decoder_self"], tgt_labels, tgt_labels,
                   f"Decoder masked self-attention: {example!r}", "Target", "Target",
                   out_dir, "09_attention_decoder_self.png"),
    plot_attention(attention["cross"], tgt_labels, src_labels,
                   f"Encoder-decoder attention: {example!r}", "Target", "Source",
                   out_dir, "10_attention_cross.png"),
    plot_positional_encoding(analysis, out_dir),
    plot_embedding_pca(analysis, token_labels, out_dir),
    plot_embedding_similarity(analysis, token_labels, out_dir),
  ]
  return paths
