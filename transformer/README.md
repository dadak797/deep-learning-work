# Transformer

Implementation of the Transformer from "Attention Is All You Need" (Vaswani et al., 2017), trained on toy string tasks.

| `--task` | Example | Training input | Default save path |
|---|---|---|---|
| `reverse` (default) | `Hello World` → `dlroW olleH` | Random characters, spaces included | `./saved_models/reverse_model.pth` |
| `word_reverse` | `Hello World` → `olleH dlroW` | Random words joined by spaces (see below) | `./saved_models/word_reverse_model.pth` |

- `word_reverse` reverses each word separated by spaces and keeps the word order and the spaces
- Characters: `a`–`z`, `A`–`Z` and space (vocab size 56 including PAD/BOS/EOS)
- Strings of length 1–50 (`MAX_LEN`), generated on the fly every step
- Both tasks use the same model architecture; each task saves to its own file

# Data Generation

Training data is generated randomly at every step, so there is no dataset file and the model never sees the same batch twice.
The validation set (4 batches × 128 strings) is generated once with a fixed seed (42), so accuracy is comparable across runs.

Each sample is encoded at the character level:

| | Tokens (`Hello World`, `word_reverse`) |
|---|---|
| `src` (encoder input) | `H e l l o ␣ W o r l d <eos>` |
| `tgt_in` (decoder input, shifted right) | `<bos> o l l e H ␣ d l r o W` |
| `tgt_out` (label) | `o l l e H ␣ d l r o W <eos>` |

- Strings in a batch are padded with `<pad>`, which is masked in attention and ignored in the loss

### `reverse`

- Total length: uniform in 1–50
- Each character: uniform over the 53 characters (letters and space)
- Target: `s[::-1]`

### `word_reverse`

`reverse`'s generator is not used here because space is 1 of 53 characters: most strings would have almost no spaces, which is the same as `reverse`.
Instead, a sentence is built from words and spaces:

1. Pick the total length uniformly in 1–50
2. Leading / trailing spaces: each added with 20% probability (1–3 spaces) (`EDGE_SPACE_PROB`)
3. Words: letters only; length 1–10 with 90% probability, 11–50 with 10% probability (`LONG_WORD_PROB`)
4. Between words: 1 space with 80% probability, 2–3 spaces with 20% probability (`MULTI_SPACE_PROB`)
5. Cut the words to the remaining length, then add the leading/trailing spaces

Target: `" ".join(word[::-1] for word in s.split(" "))`.
`split(" ")` keeps empty strings for repeated, leading and trailing spaces, so every space stays in place.

```
'zx zU K U'                                          -> 'xz Uz K U'
'JNXW jZ UYAYAV   SAEbmPvi eIfiRt KZEui CSmA VIFLt'  -> 'WXNJ Zj VAYAYU   ivPmbEAS tRifIe iuEZK AmSC tLFIV'
'XzsCWvQ GFxbljESPPQnRJeaaNmfGrdiB'                  -> 'QvWCszX BidrGfmNaaeJRnQPPSEjlbxFG'
```

In 20,000 generated samples:

| Property | Ratio |
|---|---|
| Leading space | 20% |
| Trailing space | 19% |
| Repeated spaces between words | 38% of sentences |
| More than one word | 86% of sentences |
| Words longer than 10 letters | 6.5% of words |

- Long words appear less than 10% because they are often cut by the total length
- The probabilities are constants at the top of the data section in `main.py`

# Train

```
python main.py --task reverse --steps 10000
python main.py --task word_reverse --steps 10000 --batch_size 128 --device mps
```

- The default 3000 steps is not enough for strings up to 50 characters; use about 10000 steps

- `--task`: reverse, word_reverse (default: reverse)
- `--steps`: Number of training steps (default: 3000)
- `--batch_size`: Batch size (default: 128)
- `--warmup_steps`: Warmup steps for the learning rate schedule (default: 400, paper: 4000)
- `--device`: cpu, cuda, mps(for Mac) (default: mps)
- `--save_path`: Path to save the trained weights (default: depends on `--task`, see the table above; the folder is created if missing)

# Plots

`analyze.py` saves plots from a trained model and its training log, without changing `main.py`.
It needs `matplotlib` (`pip install matplotlib`).

```
mkdir -p logs/reverse
python -u main.py --task reverse | tee logs/reverse/train.log   # train and keep the printed log
python analyze.py --task reverse                                # save plots to logs/reverse/
```

- `-u` turns off output buffering so the log appears in the terminal while training
- Without `train.log`, plots 01 and 03 are skipped and the rest are still saved
- Options: `--task`, `--save_path`, `--log_dir` (default: `./logs/<task>`), `--warmup_steps` (default: 400), `--device`

| File | Content | Source |
|---|---|---|
| `01_loss.png` | Training loss every 100 steps and the label smoothing floor (0.715) | log |
| `02_learning_rate.png` | `rate()` schedule with the values printed in the log | `rate()`, log |
| `03_val_exact_match.png` | Validation exact match every 500 steps | log |
| `04_accuracy_by_length.png` | Accuracy by input length 1–65, including lengths longer than training | model |
| `05_error_by_position.png` | Error rate by input length and output position | model |
| `06_confusion_matrix.png` | Which characters are confused with which (errors only) | model |
| `07_confidence.png` | Max softmax probability of correct and wrong predictions | model |
| `08_attention_encoder_self.png` | Encoder self-attention of every layer and head | model |
| `09_attention_decoder_self.png` | Decoder masked self-attention | model |
| `10_attention_cross.png` | Encoder-decoder attention (anti-diagonal for `reverse`) | model |
| `11_positional_encoding.png` | Positional encoding values and the similarity between positions | model |
| `12_embedding_pca.png` | Token embeddings projected to 2D with PCA | model |
| `13_embedding_similarity.png` | Cosine similarity between token embeddings (mean-centered) | model |

- Attention plots use the string `Attention Is All You Need` (`MultiHeadAttention` keeps the last attention weights in `self.attn`)
- Plots 06 and 07 use the same validation set as training (seed 42)

# Inference

- Loads the weights for `--task` (or from `--save_path`) and transforms strings typed in the terminal
- Empty input or Ctrl+D to quit

```
python main.py --task reverse --infer_mode
python main.py --task word_reverse --infer_mode --device cpu
```

`reverse`:

```
> hello
[O] 'olleh'
>  ab 
[O] ' ba '
> Hello123
Unsupported characters: '123' (only a-z, A-Z, space)
> The quick brown fox jumps over the lazy dog
[O] 'god yzal eht revo spmuj xof nworb kciuq ehT'
> abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
[X] 'ZXWVUTSRQPONMLKJIHGFEDCBAzyxwvutsrqponmlkjihgfedcba' (longer than training max 50)
```

`word_reverse`:

```
> Hello World
[O] 'olleH dlroW'
>  ab  cd 
[O] ' ba  dc '
>   Attention   is all  you need 
[O] '  noitnettA   si lla  uoy deen '
> abcdefghijklmnopqrstuvwxyz
[O] 'zyxwvutsrqponmlkjihgfedcba'
> Hi abcdefghijklmnopqrstuvwxyz there
[O] 'iH zyxwvutsrqponmlkjihgfedcba ereht'
```

- `[O]`/`[X]` shows whether the prediction matches the correct answer for `--task`
- Leading/trailing spaces are kept as part of the input, and the output is quoted so they are visible
- `word_reverse` handles leading/trailing/repeated spaces and long words because they are included in the training data (see [Data Generation](#data-generation))

# Output

- Training loss and learning rate are printed every 100 steps
- Exact-match accuracy on a fixed validation set (512 strings) is printed every 500 steps
- After training, the weights are saved to `--save_path` and a few examples are printed

`reverse`:

```
step 10000 | val exact match 99.41%
Saved model to ./saved_models/reverse_model.pth
[O] 'hello' -> 'olleh'
[O] 'Hello World' -> 'dlroW olleH'
[O] 'Attention Is All You Need' -> 'deeN uoY llA sI noitnettA'
```

`word_reverse`:

```
step 10000 | val exact match 99.61%
Saved model to ./saved_models/word_reverse_model.pth
[O] 'hello' -> 'olleh'
[O] 'Hello World' -> 'olleH dlroW'
[O] 'Attention Is All You Need' -> 'noitnettA sI llA uoY deeN'
[X] 'The quick brown fox jumps over the lazy dog and keeps running' -> 'ehT kciuq nworb xof spmuj revo eht yzal god dna eht'
```

- 10000 steps take about 7–10 minutes on an Apple Silicon Mac (MPS); accuracy reaches about 99% after 5000–6000 steps
- Loss stays around 0.7 instead of reaching 0 because of label smoothing (0.1)
- The last example (61 characters) is longer than any training string (max 50), so it usually fails

# Model

| Setting | This task | Paper (base) |
|---|---|---|
| Layers (`Nx`) | 2 | 6 |
| `d_model` | 128 | 512 |
| `d_ff` | 512 | 2048 |
| Heads | 4 | 8 |
| Dropout | 0.1 | 0.1 |
| Parameters | 0.9M | 65M |

- Post-LN encoder/decoder, sinusoidal positional encoding, embeddings scaled by `sqrt(d_model)`
- Source embedding, target embedding and the output projection share one weight matrix
- Adam (β1 = 0.9, β2 = 0.98, ε = 1e-9) with the warmup learning rate schedule (Eq. 3)
- Cross-entropy loss with label smoothing 0.1, padding ignored
- Greedy decoding for inference

To build the paper's base model for another task:

```python
model = make_model(src_vocab_size, tgt_vocab_size)  # Nx=6, d_model=512, d_ff=2048, h=8
```
