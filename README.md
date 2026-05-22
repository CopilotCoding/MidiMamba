# MidiGen3

**WIP:** MIDI generation using SM1 (Scalar Mamba 1), a novel SSM variant implemented from scratch in pure PyTorch with no custom CUDA kernels. Trains on RTX 50-series (Blackwell sm_120) where the official mamba-ssm library doesn't run. Trained on 163K MIDI files, 2.5B tokens.

Pure PyTorch, Windows-compatible, bfloat16-native. No mamba-ssm, no causal-conv1d, no Triton dependency.

---

## Installation

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install pretty_midi mido numpy rich
```

`requirements.txt` lists the full set. PyTorch 2.4+ required for bfloat16 autocast and fused AdamW.

---

## Pipeline overview

```
Raw MIDI files
    │
    ▼
midideduper.py          (optional) deduplicate raw MIDI by file hash
    │
    ▼
scan_dataset.py         Phase 1 — extract features, compute auto-buckets, write vocab_config.json
    │
    ▼
build_dataset.py        Phase 2 — tokenize all MIDI to .npy token arrays
    │
    ▼
validate_tokens.py      (optional but recommended) verify dataset health before training
    │
    ▼
train.py                Train the model
    │
    ▼
eval_generated.py       Objective quality metrics on generated samples
diagnostic.py           Conditioning response + SSM coherence tests
    │
    ▼
generate.py             Generate MIDI from a checkpoint
```

---

## Step 0 — Deduplicate raw MIDI (optional)

Remove exact duplicate MIDI files from your corpus before scanning. Keeps the oldest copy by creation time.

```
python midideduper.py --path C:\midi\corpus
```

| Flag | Default | Description |
|---|---|---|
| `--path` | required | Root directory to scan recursively |
| `--workers` | 12 | Parallel hash workers |
| `--dry-run` | off | Report duplicates without deleting |
| `--log` | dedupe_log.txt | Log file path |

---

## Step 1 — Scan dataset

Extracts features from every MIDI file, computes data-driven bucket boundaries, writes `corpus_stats.json` and `vocab_config.json`.

```
python scan_dataset.py --dirs C:\midi\corpus --out stats_out
```

| Flag | Default | Description |
|---|---|---|
| `--dirs` | required | One or more root directories, scanned recursively |
| `--out` | stats_out | Output directory for corpus_stats.json and vocab_config.json |
| `--workers` | cpu_count-1 | Parallel worker processes |
| `--limit` | 0 (all) | Cap files per directory (useful for testing) |

Output: `stats_out/corpus_stats.json`, `stats_out/vocab_config.json`

---

## Step 2 — Build dataset

Tokenizes all MIDI files to `.npy` token arrays. Uses pre-scanned features from Step 1. SHA256 deduplicates token sequences.

```
python build_dataset.py --stats stats_out --dirs C:\midi\corpus --out tokens_out
```

| Flag | Default | Description |
|---|---|---|
| `--stats` | required | stats_out directory from Step 1 |
| `--dirs` | required | Same directories as Step 1 |
| `--out` | tokens_out | Output directory for *_tokens.npy files |
| `--workers` | cpu_count-1 | Parallel worker processes |
| `--limit` | 0 (all) | Cap files per directory |
| `--chunk_size` | 500 | Files per IPC chunk |

Output: `tokens_out/*.npy`, `tokens_out/manifest.json`, `tokens_out/errors.log`

---

## Step 3 — Validate dataset (recommended)

Check token distribution, entropy, sequence lengths, and near-duplicates before spending GPU time.

```
python validate_tokens.py --data tokens_out --stats stats_out
```

| Flag | Default | Description |
|---|---|---|
| `--data` | required | tokens_out directory from Step 2 |
| `--stats` | required | stats_out directory from Step 1 |
| `--top_n` | 100 | Top N tokens to show in frequency table |
| `--max_files` | 0 (all) | Cap files to sample |

Example output on a healthy corpus:
```
Sequence stats — Files: 163,068 | Min: 131 | Avg: 15,499 | P50: 13,573 | P90: 31,029 | P99: 53,173 | Max: 150,889
Token stats    — Total: 2,527,427,619 | Unique: 441 | Entropy: 5.68 bits | Duplicates: 0
```

---

## Step 4 — Train

### Find the largest model that fits in your VRAM

```
python train.py tokens_out --stats stats_out --seq_len 4096 --batch_size 1 --sweep_models
```

Prints a table of model sizes with peak VRAM usage and a ready-to-paste training command.

### Test VRAM for a specific config

```
python train.py tokens_out --stats stats_out ^
    --d_model 704 --n_layers 10 --seq_len 4096 --batch_size 1 --test_vram
```

### Train

```
python train.py tokens_out --stats stats_out --seq_len 4096 --batch_size 1 --grad_accum 8 --d_model 704 --n_layers 10 --epochs 1 --sample_every 500 --out run
```

Auto-resumes from `run/checkpoints/latest` if it exists. Best val loss checkpoint always saved to `run/checkpoints/best`.

> **Note:** `--compile` is supported but may hang on Windows (Triton not available). Leave it off unless you're on Linux.

### Resume explicitly

```
python train.py tokens_out --stats stats_out ^
    --d_model 704 --n_layers 10 ^
    --seq_len 4096 --batch_size 1 --grad_accum 8 ^
    --resume run/checkpoints/latest
```

### All training flags

| Flag | Default | Description |
|---|---|---|
| `token_dir` | required (positional) | tokens_out directory |
| `--stats` | required | stats_out directory |
| `--out` | run | Output directory for checkpoints, logs, samples |
| `--resume` | None | Explicit checkpoint path to resume from |
| **Model** | | |
| `--d_model` | 512 | Model dimension |
| `--n_layers` | 12 | Number of layers |
| `--d_conv` | 4 | Causal conv kernel size |
| `--expand` | 2 | d_inner = d_model * expand |
| `--d_ff_mult` | 2.667 | SwiGLU expansion multiplier |
| `--dropout` | 0.1 | Dropout rate |
| `--grad_checkpoint` | off | Gradient checkpointing (saves VRAM, ~20% slower) |
| **Training** | | |
| `--seq_len` | 16384 | Max sequence length |
| `--batch_size` | 1 | Sequences per GPU step |
| `--grad_accum` | 8 | Gradient accumulation steps |
| `--epochs` | 10 | Training epochs (1 epoch = full corpus pass; on 2.5B tokens, 1–2 is typical) |
| `--lr` | 3e-4 | Peak learning rate |
| `--min_lr` | 3e-5 | Minimum LR at end of cosine schedule |
| `--warmup_frac` | 0.02 | Fraction of total steps for LR warmup |
| `--weight_decay` | 0.1 | AdamW weight decay |
| `--grad_clip` | 1.0 | Gradient norm clip |
| `--compile` | off | Attempt torch.compile (may hang on Windows) |
| **Dataset** | | |
| `--val_frac` | 0.02 | Fraction of files held out for validation |
| `--min_tokens` | 256 | Minimum sequence length to include |
| `--num_workers` | 0 | DataLoader workers (keep 0 on Windows with memmap) |
| **Checkpointing** | | |
| `--val_every` | 500 | Validate every N optimizer steps |
| `--val_batches` | 50 | Validation batches per eval |
| `--ckpt_every` | 1000 | Save latest checkpoint every N steps |
| `--ckpt_minutes` | 30 | Also save a timestamped checkpoint every N minutes |
| `--sample_every` | 0 | Auto-generate a sample MIDI every N steps (0 = off) |
| **Utility modes** | | |
| `--sweep_models` | off | VRAM sweep across model sizes, then exit |
| `--test_vram` | off | Single forward+backward VRAM test, then exit |

---

## Step 5 — Generate

### Basic generation (unconditioned)

```
python generate.py run/checkpoints/best output.mid
```

### Conditioned generation

```
python generate.py run/checkpoints/best output.mid ^
    --tempo 120 --key_root 0 --key_minor 0 --has_drums 1
```

### Batch — N variations in parallel

```
python generate.py run/checkpoints/best output.mid --batch 4
```

Writes `output_00.mid`, `output_01.mid`, etc.

### All generation flags

| Flag | Default | Description |
|---|---|---|
| `checkpoint` | required (positional) | Checkpoint directory |
| `output` | required (positional) | Output .mid path (stem for --batch > 1) |
| `--batch` | 1 | Generate N variations in parallel |
| `--max_tokens` | 50000 | Maximum tokens to generate |
| `--min_tokens` | 4000 | Minimum tokens before EOS is allowed |
| `--temperature` | 0.92 | Sampling temperature |
| `--top_p` | 0.93 | Nucleus sampling cutoff |
| `--seed` | None | Random seed for reproducibility |
| `--rep_penalty` | 1.08 | Pitch repetition penalty (1.0 = off) |
| `--rep_window` | 256 | Token window for repetition penalty |

### Conditioning flags (all optional)

| Flag | Type | Description |
|---|---|---|
| `--tempo` | float | BPM |
| `--duration_sec` | float | Target duration in seconds |
| `--n_bars` | int | Target bar count |
| `--pitch_min` | int | Lowest pitch (21–108) |
| `--pitch_max` | int | Highest pitch (21–108) |
| `--pitch_range` | int | Pitch span in semitones |
| `--note_density` | float | Notes per bar |
| `--avg_dur_sec` | float | Average note duration in seconds |
| `--polyphony` | float | Average simultaneous notes |
| `--rest_density` | float | Fraction of time silent (0–1) |
| `--total_notes` | int | Total note count |
| `--ioi_cv` | float | Rhythmic irregularity (0–3) |
| `--pitch_variety` | float | Pitch variety (0–1) |
| `--interval_diversity` | float | Average melodic interval size |
| `--ts_num` | int | Time signature numerator |
| `--ts_den` | int | Time signature denominator |
| `--key_root` | int | 0=C, 1=C#, 2=D, 3=Eb, 4=E, 5=F, 6=F#, 7=G, 8=Ab, 9=A, 10=Bb, 11=B |
| `--key_minor` | int | 0 = major, 1 = minor |
| `--key_detected` | int | 1 = key signature present |
| `--n_tracks` | int | Number of instrument tracks |
| `--has_drums` | int | 1 = drums, 0 = no drums |
| `--midi_format` | int | MIDI format: 0 or 1 |

---

## Evaluation and diagnostics

### Objective quality metrics

```
python eval_generated.py run/checkpoints/best --stats stats_out --n 20
```

Generates N samples and computes duplicate note %, n-gram repeat rate, unique token ratio, pitch entropy, note density, bar similarity.

| Flag | Default | Description |
|---|---|---|
| `checkpoint` | required | Checkpoint directory |
| `--stats` | required | stats_out directory |
| `--n` | 20 | Number of samples to evaluate |
| `--max_tokens` | 8000 | Tokens per sample |
| `--temperature` | 0.92 | Sampling temperature |
| `--top_p` | 0.93 | Nucleus sampling cutoff |

### Conditioning response diagnostic

```
python diagnostic.py --checkpoint run/checkpoints/best --vocab stats_out/vocab_config.json
```

Tests whether the model responds to conditioning (tempo, key, drums). Also checks for conditioning token leakage.

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | run/checkpoints/best | Checkpoint directory |
| `--vocab` | stats_out/vocab_config.json | vocab_config.json path |
| `--tokens` | 600 | Tokens per test generation |

### Profile training step

```
python profile_step.py --seq_len 4096 --d_model 704 --n_layers 10 --stats stats_out
```

Runs torch.profiler and prints ops sorted by CUDA time.

| Flag | Default | Description |
|---|---|---|
| `--seq_len` | 53178 | Sequence length to profile |
| `--d_model` | 512 | Model dimension |
| `--n_layers` | 16 | Number of layers |
| `--grad_checkpoint` | on | Use gradient checkpointing |
| `--steps` | 6 | Steps to profile |
| `--stats` | stats_out | stats_out directory |

---

## Testing

```
python test_scan.py    # correctness — all cases should print [PASS]
python bench_scan.py   # correctness + gradient check + throughput
```

---

## Output structure

```
run/
  checkpoints/
    best/             — best validation loss checkpoint
      model.pt
      optimizer.pt
      meta.json
    latest/           — most recent checkpoint (auto-resume target)
    step_0001000/     — timed permanent checkpoints
  samples/            — auto-generated MIDI (if --sample_every > 0)
  loss_log.csv        — step, train_loss, val_loss, lr
```

---

## Architecture

**SSM:** SM1 (Scalar Mamba 1) — a Mamba1 variant with d_state=1. The state per feature is a single scalar, enabling an exact closed-form scan via cumprod + cumsum: two vectorized GPU ops, no Python loop, no OOM risk, numerically stable. SM1 is the one of the only Mamba formulation that admits this closed-form without custom CUDA kernels, making it viable on Blackwell (sm_120) where mamba-ssm is unsupported.

**Scan:**
```
L[t]     = cumprod(dA, dim=1)           cumulative decay
h[t]     = L[t] * cumsum(dBx/L, dim=1) hidden states  
y[t]     = h[t] * C[t]                 readout
```

**Tokenizer:** BAR / SECTION / TEMPO / POS / TRACK / PITCH / DUR / VEL schema. Conditioning prefix prepended to every sequence with data-driven bucket boundaries from corpus statistics. 442 total tokens.

**Training:** bfloat16 autocast (native on sm_86+). Fused AdamW. Cosine LR with warmup. Memmap dataset — no full corpus preload. Songs longer than `seq_len` are split into chunks; the SSM hidden state is threaded across chunks of the same song and reset only at song boundaries. Gradients flow within each chunk only (TBPTT). Shuffle happens at song level so chunks always arrive in order.

**Inference:** Recurrent step mode — O(1) memory and compute per token, no context window limit.
