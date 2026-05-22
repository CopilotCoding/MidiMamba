# MidiGen3

MIDI generation using a correct Mamba1 SSM. Pure PyTorch, Windows-compatible, bfloat16-native on RTX 30/40/50 series.

Built from midigen2 with the core architecture bug fixed (x_ssm was a scalar per head — every feature in a head was identical, wasted capacity), bfloat16 training, per-batch dynamic padding, memmap dataset, and all utility scripts intact.

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
python scan_dataset.py --dirs C:\midi\bach C:\midi\maestro C:\midi\giantmidi --out stats_out
```

| Flag | Default | Description |
|---|---|---|
| `--dirs` | required | One or more root directories, scanned recursively |
| `--out` | stats_out | Output directory for corpus_stats.json and vocab_config.json |
| `--workers` | cpu_count-1 | Parallel worker processes |
| `--limit` | 0 (all) | Cap files per directory (useful for testing) |

Output: `stats_out/corpus_stats.json`, `stats_out/vocab_config.json`

Prints a full corpus diversity report: key distribution, tempo histogram, bucket occupancy warnings, duplicate estimate.

---

## Step 2 — Build dataset

Tokenizes all MIDI files to `.npy` token arrays. Uses pre-scanned features from Step 1 (no mido re-extraction). SHA256 deduplicates token sequences. Trims extreme outliers (sequences > 5x P99 length).

```
python build_dataset.py --stats stats_out --dirs C:\midi\bach C:\midi\maestro C:\midi\giantmidi --out tokens_out
```

| Flag | Default | Description |
|---|---|---|
| `--stats` | required | stats_out directory from Step 1 |
| `--dirs` | required | Same directories as Step 1 |
| `--out` | tokens_out | Output directory for *_tokens.npy files |
| `--workers` | cpu_count-1 | Parallel worker processes |
| `--limit` | 0 (all) | Cap files per directory |
| `--chunk_size` | 500 | Files per IPC chunk (tune for Windows IPC overhead) |

Output: `tokens_out/*.npy`, `tokens_out/manifest.json`, `tokens_out/errors.log`

---

## Step 3 — Validate dataset (recommended)

Check token distribution, entropy, sequence lengths, conditioning coverage, and near-duplicates before spending GPU time on a bad dataset.

```
python validate_tokens.py --data tokens_out --stats stats_out
```

| Flag | Default | Description |
|---|---|---|
| `--data` | required | tokens_out directory from Step 2 |
| `--stats` | required | stats_out directory from Step 1 |
| `--top_n` | 100 | How many top tokens to show in frequency table |
| `--max_files` | 0 (all) | Cap files to sample |

Watch for: low entropy warnings, skewed conditioning dimensions, dead tokens > 30%, near-duplicate rate > 20%.

---

## Step 4 — Train

### Find the largest model that fits in your VRAM

Run the sweep before committing to a config. Tests increasing model sizes with a real forward+backward pass and reports peak VRAM.

```
python train.py tokens_out --stats stats_out --seq_len 16384 --batch_size 1 --sweep_models
```

Prints a table and outputs a ready-to-paste training command at the bottom.

### Test VRAM for a specific config

```
python train.py tokens_out --stats stats_out --d_model 768 --n_layers 16 --seq_len 16384 --batch_size 1 --test_vram
```

### Train

```
python train.py tokens_out --stats stats_out --d_model 768 --n_layers 16 --seq_len 16384 --batch_size 1 --grad_accum 8 --epochs 10 --out run
```

Auto-resumes from `run/checkpoints/latest` if it exists. Best val loss checkpoint always saved to `run/checkpoints/best`.

### Resume explicitly

```
python train.py tokens_out --stats stats_out --d_model 768 --n_layers 16 --seq_len 16384 --batch_size 1 --grad_accum 8 --resume run/checkpoints/latest
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
| `--d_state` | 32 | SSM state size per feature dim |
| `--d_conv` | 4 | Causal conv kernel size |
| `--expand` | 2 | d_inner = d_model * expand |
| `--d_ff_mult` | 2.667 | SwiGLU expansion multiplier |
| `--dropout` | 0.1 | Dropout rate |
| `--grad_checkpoint` | off | Gradient checkpointing (saves VRAM, slows training ~20%) |
| **Training** | | |
| `--seq_len` | 16384 | Max sequence length |
| `--batch_size` | 1 | Sequences per GPU step |
| `--grad_accum` | 8 | Gradient accumulation steps (effective batch = batch_size * grad_accum) |
| `--epochs` | 10 | Training epochs |
| `--lr` | 3e-4 | Peak learning rate |
| `--min_lr` | 3e-5 | Minimum LR at end of cosine schedule |
| `--warmup_frac` | 0.02 | Fraction of total steps used for LR warmup |
| `--weight_decay` | 0.1 | AdamW weight decay |
| `--grad_clip` | 1.0 | Gradient norm clip |
| `--compile` | off | Attempt torch.compile (Windows: may silently fall back to eager) |
| **Dataset** | | |
| `--val_frac` | 0.02 | Fraction of files held out for validation |
| `--min_tokens` | 256 | Minimum sequence length to include |
| `--num_workers` | 0 | DataLoader workers (keep 0 on Windows with memmap) |
| **Checkpointing** | | |
| `--val_every` | 500 | Validate every N optimizer steps |
| `--val_batches` | 50 | Number of validation batches per eval |
| `--ckpt_every` | 1000 | Save latest checkpoint every N steps |
| `--ckpt_minutes` | 30 | Also save a timestamped checkpoint every N minutes |
| `--sample_every` | 0 | Auto-generate a sample MIDI every N steps (0 = off) |
| **Utility modes** | | |
| `--sweep_models` | off | VRAM sweep across model sizes, then exit |
| `--test_vram` | off | Single forward+backward VRAM test, then exit |

### Recommended configs by VRAM

These are starting points. Run `--sweep_models` to confirm on your hardware.

| VRAM | d_model | n_layers | ~Params | seq_len | batch | grad_accum |
|---|---|---|---|---|---|---|
| 8 GB | 512 | 12 | ~85M | 8192 | 1 | 8 |
| 16 GB | 768 | 16 | ~245M | 16384 | 1 | 8 |
| 24 GB | 768 | 20 | ~305M | 16384 | 2 | 4 |
| 24 GB | 896 | 20 | ~415M | 16384 | 1 | 8 |

---

## Step 5 — Generate

### Basic generation (unconditioned)

```
python generate.py run/checkpoints/best output.mid
```

### Conditioned generation

```
python generate.py run/checkpoints/best output.mid --tempo 120 --key_root 0 --key_minor 0 --has_drums 1 --note_density 8
```

### Batch generation — N variations in parallel

```
python generate.py run/checkpoints/best output.mid --batch 4
```

Writes `output_00.mid`, `output_01.mid`, `output_02.mid`, `output_03.mid`.

### Long piece

```
python generate.py run/checkpoints/best output.mid --max_tokens 80000 --min_tokens 10000 --temperature 0.95
```

### All generation flags

| Flag | Default | Description |
|---|---|---|
| `checkpoint` | required (positional) | Checkpoint directory (contains model.pt + meta.json) |
| `output` | required (positional) | Output .mid path (stem for --batch > 1) |
| `--batch` | 1 | Generate N variations in parallel on GPU |
| `--max_tokens` | 50000 | Maximum tokens to generate |
| `--min_tokens` | 4000 | Minimum tokens before EOS is allowed |
| `--temperature` | 0.92 | Sampling temperature (higher = more random) |
| `--top_p` | 0.93 | Nucleus sampling cutoff |
| `--seed` | None | Random seed for reproducibility |
| `--rep_penalty` | 1.08 | Pitch repetition penalty (1.0 = off) |
| `--rep_window` | 256 | Token window for repetition penalty |

### Conditioning flags (all optional — omitting uses neutral mid-bucket)

| Flag | Type | Description |
|---|---|---|
| `--tempo` | float | BPM (e.g. 120.0) |
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
| `--ioi_cv` | float | Rhythmic irregularity (0–3, higher = more irregular) |
| `--pitch_variety` | float | Pitch variety (0–1) |
| `--interval_diversity` | float | Average melodic interval size |
| `--ts_num` | int | Time signature numerator (e.g. 4) |
| `--ts_den` | int | Time signature denominator (e.g. 4) |
| `--key_root` | int | Key root: 0=C, 1=C#, 2=D, 3=Eb, 4=E, 5=F, 6=F#, 7=G, 8=Ab, 9=A, 10=Bb, 11=B |
| `--key_minor` | int | 0 = major, 1 = minor |
| `--key_detected` | int | 1 = key signature present in source data |
| `--n_tracks` | int | Number of instrument tracks |
| `--has_drums` | int | 1 = drums present, 0 = no drums |
| `--midi_format` | int | MIDI format: 0 or 1 |

---

## Evaluation and diagnostics

### Objective quality metrics

Generates N samples and computes: duplicate note %, n-gram repeat rate, unique token ratio, pitch entropy, note density, bar-to-bar similarity. Flags models with repetition or monotone output.

```
python eval_generated.py run/checkpoints/best --stats stats_out --n 20
```

| Flag | Default | Description |
|---|---|---|
| `checkpoint` | required (positional) | Checkpoint directory |
| `--stats` | required | stats_out directory |
| `--n` | 20 | Number of samples to generate and evaluate |
| `--max_tokens` | 8000 | Tokens per sample |
| `--temperature` | 0.92 | Sampling temperature |
| `--top_p` | 0.93 | Nucleus sampling cutoff |

### Conditioning response + SSM coherence diagnostic

Tests whether the model actually responds to conditioning (e.g. high tempo vs low tempo, major vs minor, drums vs no drums). Also checks for conditioning token leakage into the music section.

```
python diagnostic.py --checkpoint run/checkpoints/best --vocab stats_out/vocab_config.json
```

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | run/checkpoints/best | Checkpoint directory |
| `--vocab` | stats_out/vocab_config.json | vocab_config.json path |
| `--tokens` | 600 | Tokens per test generation |

### Profile training step (find bottlenecks)

Runs torch.profiler for N steps at a given seq_len and prints ops sorted by CUDA time. Compare seq_len=8192 vs seq_len=32768 to find what grows superlinearly.

```
python profile_step.py --seq_len 16384 --d_model 768 --n_layers 16 --stats stats_out
```

| Flag | Default | Description |
|---|---|---|
| `--seq_len` | 53178 | Sequence length to profile |
| `--d_model` | 512 | Model dimension |
| `--n_layers` | 16 | Number of layers |
| `--grad_checkpoint` | on | Use gradient checkpointing |
| `--no_checkpoint` | — | Disable gradient checkpointing |
| `--steps` | 6 | Number of steps to profile |
| `--stats` | stats_out | stats_out directory (for vocab size) |

---

## Testing

### SSM scan correctness

Verifies `_ssm_scan` against a sequential reference loop at multiple sequence lengths. Run this after any changes to model.py.

```
python test_scan.py
```

All cases should print `[PASS]`. If any print `[FAIL]` the scan math is broken.

### SSM scan benchmark

Correctness + gradient check + wall-clock throughput. Useful for confirming performance on new hardware.

```
python bench_scan.py
```

---

## Output structure

After training:

```
run/
  checkpoints/
    best/           — best validation loss checkpoint
      model.pt
      optimizer.pt
      meta.json
    latest/         — most recent checkpoint (auto-resume target)
    step_0001000/   — timed permanent checkpoints
    step_0002000/
  samples/          — auto-generated MIDI samples (if --sample_every > 0)
    step_0001000.mid
  loss_log.csv      — step, train_loss, val_loss, lr
```

`meta.json` contains everything needed to reload the model:

```json
{
  "step": 12500,
  "epoch": 3,
  "best_val": 1.8432,
  "vocab_config": "stats_out/vocab_config.json",
  "model_cfg": {
    "vocab_size": 442,
    "d_model": 768,
    "n_layers": 16,
    ...
  }
}
```

---

## Architecture notes

**Model:** Mamba1 SSM. `A_log`, `dt_bias`, `D` are `(d_inner,)` — one parameter per feature dimension. State per layer per batch element is `(d_inner, d_state)`. No multi-head grouping.

**SSM scan:** Hillis-Steele parallel prefix scan, chunked across sequence for memory efficiency. CHUNK=2048 means ~8 sequential carry iterations at seq_len=16384. fp32 accumulation, cast back to input dtype on return.

**Tokenizer:** BAR / SECTION / TEMPO / POS / TRACK / PITCH / DUR / VEL token schema. Conditioning prefix prepended to every sequence — data-driven bucket boundaries computed from corpus statistics, not hardcoded. Token layout is identical to midigen2; existing tokenized datasets are compatible.

**Training:** bfloat16 autocast (native on sm_86+, including RTX 5060 Ti sm_100). Fused AdamW. Cosine LR with warmup. Per-batch dynamic padding via `PadCollator` — sequences padded to longest in batch, not to global max_seq. Memmap dataset — no full corpus preload.

**Inference:** Recurrent step mode — O(1) memory and compute per token regardless of sequence length. No context window limit.
