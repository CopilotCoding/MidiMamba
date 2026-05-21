# MidiMamba — Long-Context MIDI Generation

Mamba-based MIDI generation model designed for full-song coherence at 50K+ token context.
No sliding window. No repetition collapse. Linear memory scaling.

Pure PyTorch — no external CUDA compilation, works on Windows and Linux.

---

## Why Mamba?

Standard transformers use quadratic attention — at 50K tokens, memory explodes.
Mamba's SSM state carries full history in a fixed-size recurrent state at O(1) memory per step.
The model can generate hour-long pieces without ever losing context of what it already played.

Two modes:
- **Training** — segmented log-space cumsum scan over full sequence, pure CUDA ops
- **Inference** — single-token recurrent step, O(1) memory regardless of sequence length

### What makes this implementation different

- **53K context on a small model.** Standard practice scales context and parameters together. This project runs extreme context on a small model, forcing compression rather than memorization — closer to how biological memory works than most ML systems.
- **Pure PyTorch SSM.** No mamba-ssm, no bitsandbytes, no flash-attention. Everything is PyTorch primitives. Runs anywhere PyTorch runs, on any OS, on consumer hardware.
- **Data quality over quantity.** SHA256 dedup at file level and token sequence level, 12-byte pre-filter before any parsing, scanner-derived rejection criteria. Cleaner than most published MIDI generation datasets.
- **Conditioning vocab built from scan output.** Bucket boundaries represent real percentiles of your actual corpus, not hardcoded ranges.
- **No fallback parsing.** Files not in corpus_stats.json are skipped entirely — no silent degradation.

---

## Pipeline

```
1. scan_dataset.py    — pre-filter + scan MIDI files, build auto-bucketed vocab config
2. build_dataset.py   — tokenize using scan results, deduplicate, trim outliers
3. validate_tokens.py — validate token distribution, near-dupes, dead tokens
4. train.py           — train with hash-based train/val split (no leakage)
5. generate.py        — generate MIDI with optional conditioning
6. eval_generated.py  — objective quality metrics on generated samples
```

---

## Setup

Install PyTorch first (preserve your CUDA build):
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA:
```
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Install the rest:
```
pip install -r requirements.txt
```

---

## Data Preparation

**Deduplicate your raw MIDI files first.** Datasets like GiantMIDI contain many files with
different filenames but identical byte content — the same transcription saved under multiple
names. Training on raw GiantMIDI means certain pieces get over-represented at 3-10x without
any visible indication.

Use the included `midideduper.py` to SHA256-deduplicate your raw files, keeping the oldest
copy per hash:

```
# Dry run first — see what would be deleted
python midideduper.py --path "path/to/midi" --dry-run

# Actually delete duplicates
python midideduper.py --path "path/to/midi"
```

After deduplication, GiantMIDI typically loses 15-25% of its files. A log of all kept and
deleted files is written to `dedupe_log.txt`.

On Windows, `rglob("*.mid")` and `rglob("*.MID")` both match the same files on
case-insensitive NTFS. The scanner and tokenizer both deduplicate by lowercase path.

---

## Step 1: Scan Dataset

Scans all MIDI folders, extracts every measurable feature, and auto-computes data-driven
bucket boundaries for each conditioning dimension.

### Pre-filter (12 bytes, no parsing)

Every file is rejected before any MIDI library opens it if:
- File size outside 512 bytes – 20MB
- Double extensions (`.mid.mid`)
- Magic bytes are not `MThd`
- Header chunk size is not exactly 6
- MIDI format is not 0, 1, or 2
- Track count is 0 or >256
- Format 0 has more than 1 track; format 1 has fewer than 2

### Feature extraction

Uses mido for raw MIDI parsing — no piano roll, no beat tracking. All features computed
directly from note events via sweep line arithmetic and O(log N) tempo conversion.

- **Note pairing** — LIFO stack per (channel, note) pair, correct for overlapping note-ons
- **Tempo** — duration-weighted average BPM, O(log N) binary search via tempo index
- **key_detected** — distinguishes real key signatures from C major defaults
- **has_drums** — detected via channel 9; conditioning token allows drum-free generation

```
python scan_dataset.py --dirs "path/to/midi" --out stats_out
```

Outputs:
- `stats_out/corpus_stats.json` — per-file feature vectors (used by tokenizer)
- `stats_out/vocab_config.json` — auto-bucketed vocab config

---

## Step 2: Tokenize

Reads features directly from `corpus_stats.json` — no re-scanning. Files not in
corpus_stats.json are skipped entirely.

- **SHA256 deduplication** — token-sequence duplicates removed after file-level dedup
- **Outlier trimming** — sequences longer than 5x P99 deleted
- **Structured error logging** — all failures to `tokens_out/errors.log`

```
python build_dataset.py --stats stats_out --dirs "path/to/midi" --out tokens_out
```

---

## Step 3: Validate

Run before training. If anything shows RED, fix the data first.

```
python validate_tokens.py --data tokens_out --stats stats_out
```

Reports token entropy, conditioning coverage, dead tokens, near-duplicate rate, and
duplicate sequences. Near-duplicate warnings at 60-70% are normal for stylistically
coherent corpora and do not indicate a data problem if exact duplicates are zero.

---

## Step 3.5: Find Your Optimal Model Size

### Model sweep

Tests model configs across a range of parameter counts at your target sequence length.
Stops at first OOM, prints recommended config and exact training command.

```
python train.py tokens_out --stats stats_out --sweep_models --seq_len 53178 --batch_size 1 --grad_accum 16
```

### Single VRAM test

```
python train.py tokens_out --stats stats_out --test_vram --seq_len 53178 --d_model 160 --n_layers 6 --n_heads 4 --batch_size 1
```

### Example sweep results at seq_len=53178, batch=1, 16GB VRAM

Tested on:
- **CPU:** Intel Core i9-12900K (16 physical / 24 logical cores)
- **RAM:** 32GB DDR5
- **GPU:** NVIDIA GeForce RTX 5060 Ti (16GB VRAM)
- **Storage:** NVMe SSD
- **Motherboard:** ASUS TUF Gaming B760-PLUS WIFI

| Config | Params | VRAM | % | Headroom |
|--------|--------|------|---|----------|
| d=64  L=4  H=2 | 0.4M |  2396MB | 14.7% | 13915MB |
| d=128 L=6  H=4 | 1.9M |  5976MB | 36.6% | 10335MB |
| d=160 L=6  H=4 | 3.3M |  6982MB | 42.8% |  9328MB |
| d=160 L=8  H=4 | 3.7M |  9086MB | 55.7% |  7224MB |
| d=256 L=8  H=4 | 9.7M | 13291MB | 81.5% |  3020MB |
| d=256 L=12 H=4 | 14.5M | OOM | — | — |

### Choosing seq_len

P99 of your corpus is a good target — 99% of songs fit as complete sequences.
Reduce seq_len if the smallest sweep config OOMs; increase if all configs have large headroom.

---

## Step 4: Train

### Recommended configuration for 16GB VRAM at seq_len=53178

```
python train.py tokens_out --stats stats_out ^
  --seq_len 53178 --batch_size 1 --grad_accum 16 ^
  --d_model 160 --n_layers 6 --n_heads 4
```

**Why this config:**
- `seq_len=53178` — P99 of a typical corpus. Full musical arc visible at every training step.
- `batch_size=1` — seq_len=53K saturates VRAM at batch=1.
- `grad_accum=16` — simulates batch_size=16. Each optimizer step sees ~850K tokens.
- `d_model=160 n_layers=6 n_heads=4` — 3.3M parameters. Stable, ~43% VRAM at this seq_len.
- 1 epoch over ~160K sequences = ~10K optimizer steps, ~19-21 hours on a mid-range consumer GPU.

**Training regime: small model + enormous context**

With 2.5B tokens and 3.3M parameters you are massively token-rich relative to parameter
count (Chinchilla optimal would be ~66M tokens for this model size). The model cannot
memorize — it is forced to compress musical structure. The 53K context window means it
sees entire pieces, not fragments.

Past the Chinchilla compute-optimal point, loss still decreases and sample quality can
still improve — especially for generative music where long-context structure, dataset
diversity, and tokenizer quality dominate parameter scaling. Whether this produces
generalization or sophisticated pattern-matching is what the val loss curve reveals.

**What to watch:**
- Val loss first appears at step 250 (`--val_every 250`)
- `run/checkpoints/best/` saved whenever val loss improves
- `run/checkpoints/latest/` overwritten every 100 steps
- `run/samples/` receives a generated MIDI every 500 steps
- `run/loss_log.csv` tracks train loss, val loss, and LR per step

**Expected val loss trajectory:**
- Step 250-500: ~2.0-2.5 — model knows tokens exist, no musical logic
- Step 1000-2000: ~1.6-1.9 — rhythmic patterns emerging, bar structure understood
- Step 3000-5000: ~1.3-1.6 — phrase-level structure, conditioning tokens doing real work
- Step 6000-8000: ~1.1-1.4 — diminishing returns, long-range coherence test
- Best checkpoint likely between step 5000-8000

**Resuming:** Auto-resumes from `run/checkpoints/latest/` on restart. Train/val split is
deterministic (filename hash) — resume is safe with no data leakage.

### Training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--seq_len` | 53178 | Sequence length |
| `--batch_size` | 1 | Batch size |
| `--grad_accum` | 16 | Gradient accumulation steps |
| `--d_model` | 512 | Model dimension |
| `--n_layers` | 16 | Number of layers |
| `--n_heads` | 8 | SSM heads |
| `--d_state` | 64 | SSM state dimension |
| `--epochs` | 1 | Training epochs |
| `--lr` | 2e-4 | Peak learning rate |
| `--grad_checkpoint` | off | Trade ~33% speed for VRAM savings |
| `--compile` | off | torch.compile (slow first step, faster after) |
| `--val_every` | 250 | Validation interval |
| `--ckpt_every` | 100 | Checkpoint interval |
| `--sample_every` | 500 | Auto-generate MIDI sample interval |

### Training display

Rich live panel on a background thread — updates every 0.25 seconds regardless of main
thread blocking. Shows step/total, loss, val loss, LR, elapsed, ETA, tok/s, batch
progress, best val, and status. ETA computed from rolling 50-step average.

---

## Step 5: Generate

```
python generate.py run/checkpoints/best output.mid --max_tokens 10000
```

Conditioning examples:

```
# Specific key and tempo
python generate.py run/checkpoints/best output.mid --tempo 120 --key_root 0 --key_minor 0 --key_detected 1

# No drums
python generate.py run/checkpoints/best output.mid --has_drums 0

# Slow, dense, minor
python generate.py run/checkpoints/best output.mid --tempo 60 --key_minor 1 --polyphony 3.0

# Rhythmically loose, fast, major
python generate.py run/checkpoints/best output.mid --tempo 160 --key_minor 0 --ioi_cv 1.5
```

Generation uses single-token recurrent stepping — O(1) memory per token, no context limit.
EOS is ignored until `--min_tokens` (default 4000) are generated.

### Generation flags

| Flag | Default | Description |
|------|---------|-------------|
| `--max_tokens` | 50000 | Maximum tokens to generate |
| `--min_tokens` | 4000 | Minimum before EOS is respected |
| `--temperature` | 1.0 | Sampling temperature |
| `--top_p` | 0.92 | Nucleus sampling cutoff |
| `--rep_penalty` | 1.1 | Repetition penalty (1.0 = disabled) |
| `--rep_window` | 200 | Token window for repetition penalty |

---

## Step 6: Evaluate

```
python eval_generated.py run/checkpoints/best --stats stats_out --n 20
```

| Metric | Bad sign |
|--------|----------|
| Duplicate note % | >15% |
| 4-gram repeat rate | >0.6 |
| Unique token ratio | <0.08 |
| Pitch entropy | <1.5 bits |
| Notes per bar | <0.5 |
| Bar similarity | >0.85 |

---

## Conditioning Dimensions

281 total conditioning tokens. All optional — omitted dimensions use neutral mid-bucket.

| Flag | Description |
|------|-------------|
| `--tempo` | BPM |
| `--duration_sec` | Target duration in seconds |
| `--n_bars` | Target bar count |
| `--pitch_min` | Lowest pitch (MIDI 0-127) |
| `--pitch_max` | Highest pitch (MIDI 0-127) |
| `--pitch_range` | Pitch span in semitones |
| `--note_density` | Average notes per bar |
| `--avg_dur_sec` | Average note duration in seconds |
| `--polyphony` | Average simultaneous notes |
| `--rest_density` | Fraction of time silent (0-1) |
| `--ioi_cv` | Rhythmic irregularity (0=metronomic, 2+=chaotic) |
| `--pitch_variety` | Unique pitches / 128 MIDI range (0-1) |
| `--interval_diversity` | Average melodic interval in semitones |
| `--ts_num` | Time signature numerator |
| `--ts_den` | Time signature denominator |
| `--key_root` | Key root: 0=C 1=C# 2=D ... 11=B |
| `--key_minor` | 0=major 1=minor |
| `--key_detected` | 1=real key signature present, 0=unknown |
| `--n_tracks` | Number of instrument tracks |
| `--has_drums` | 1=drums on channel 9, 0=no drums |

---

## Output Structure

```
run/
├── checkpoints/
│   ├── latest/           — overwritten every 100 steps
│   ├── best/             — lowest val loss seen during training
│   └── step_XXXXXXX/     — permanent timed checkpoints (every 30 min)
├── samples/
│   └── step_XXXXXXX.mid  — auto-generated every 500 steps
└── loss_log.csv          — step, train_loss, val_loss, lr

tokens_out/
├── *_tokens.npy          — tokenized song files
├── manifest.json         — file index
├── errors.log            — tokenization errors
├── .split_cache.pkl      — filename→split+length cache (permanent)
├── .chunk_cache_train.pkl — chunk list cache (rebuilds on seq_len change)
└── .chunk_cache_val.pkl

stats_out/
├── corpus_stats.json     — per-file feature vectors (used by tokenizer)
└── vocab_config.json     — conditioning token config
```

---

## Architecture

### Token layout

```
[0 .. COND_END)       conditioning tokens (auto-sized from corpus)
COND_END + 0          PAD
COND_END + 1          BOS
COND_END + 2          EOS
COND_END + 3          BAR
COND_END + 4..19      POS_0..POS_15 (16th note positions within bar)
COND_END + 20..28     TRACK_0..TRACK_8
COND_END + 29..116    PITCH_21..PITCH_108 (88 piano keys)
COND_END + 117..132   DUR_1..DUR_16
COND_END + 133..140   VEL_1..VEL_8
COND_END + 141..157   TEMPO_40..TEMPO_200 (step 10)
COND_END + 158..160   SECTION_EARLY / SECTION_MID / SECTION_LATE
```

Each note encodes as: `TRACK + POS + PITCH + DUR + VEL` (5 tokens minimum).

### SSM formulation (per head)

```
h[t] = dA[t] * h[t-1] + dB[t] * x_ssm[t]   state update
y[t] = C[t] @ h[t]                            output
```

- `dA` — scalar decay per head, values in (0,1)
- `dB`, `C` — d_state vectors per head
- `x_ssm` — scalar input per head
- State size fixed at `(n_heads, d_state)` — does not grow with sequence length

### Scan implementation

Segmented log-space cumsum — ~7 Python iterations for T=53K (SEG=8192). All heavy
work inside each segment is pure CUDA (log, cumsum, exp). Numerically stable via
log-domain with clamping. Runs in float32 internally regardless of autocast.

Within each segment:
```
logcumA      = cumsum(log(dA))           # accumulated decay in log space
inv_cumA     = exp(-logcumA_prev)        # inverse for input normalization
Bu_norm      = dBx * inv_cumA           # normalize inputs
h            = cumA * cumsum(Bu_norm)   # reconstruct hidden states
             + cumA * carry             # plus propagated carry from previous segment
```

### Inference

Single-token recurrent step updates the SSM state in O(1) memory and time:
```
h_new = dA * h_prev + dB * x_ssm
y     = C @ h_new
```

No context window limit — generate indefinitely without forgetting earlier context.
