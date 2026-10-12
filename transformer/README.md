# Transformer

Implementation of the Transformer from "Attention Is All You Need" (Vaswani et al., 2017), trained on a toy string reversal task.

- Input: `Hello World` → Output: `dlroW olleH`
- Characters: `a`–`z`, `A`–`Z` and space (vocab size 56 including PAD/BOS/EOS)
- Random strings of length 1–50 (`MAX_LEN`), generated on the fly every step

# Train

```
python main.py --steps 10000
python main.py --steps 10000 --batch_size 128 --device mps
```

- The default 3000 steps is not enough for strings up to 50 characters; use about 10000 steps

- `--steps`: Number of training steps (default: 3000)
- `--batch_size`: Batch size (default: 128)
- `--warmup_steps`: Warmup steps for the learning rate schedule (default: 400, paper: 4000)
- `--device`: cpu, cuda, mps(for Mac) (default: mps)
- `--save_path`: Path to save the trained weights (default: `./saved_models/reverse_model.pth`, the folder is created if missing)

# Inference

- Loads the weights from `--save_path` and reverses strings typed in the terminal
- Empty input or Ctrl+D to quit

```
python main.py --infer_mode
python main.py --infer_mode --save_path ./saved_models/reverse_model.pth --device cpu
```

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

- `[O]`/`[X]` shows whether the prediction matches the true reversed string
- Leading/trailing spaces are kept as part of the input, and the output is quoted so they are visible

# Output

- Training loss and learning rate are printed every 100 steps
- Exact-match accuracy on a fixed validation set (512 strings) is printed every 500 steps
- After training, the weights are saved to `--save_path` and a few example reversals are printed

```
step 10000 | val exact match 99.41%
Saved model to ./saved_models/reverse_model.pth
[O] 'hello' -> 'olleh'
[O] 'Hello World' -> 'dlroW olleH'
[O] 'Attention Is All You Need' -> 'deeN uoY llA sI noitnettA'
[X] 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' -> 'ZXWVUTSRQPONMLKJIHGFEDCBAzyxwvutsrqponmlkjihgfedcba'
```

- 10000 steps take about 7 minutes on an Apple Silicon Mac (MPS); accuracy reaches about 99% after 6000 steps
- Loss stays around 0.7 instead of reaching 0 because of label smoothing (0.1)
- The last example (52 characters) is longer than any training string (max 50), so it usually fails

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
