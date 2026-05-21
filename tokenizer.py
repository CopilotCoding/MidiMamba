"""
tokenizer.py

Encodes MIDI files to token sequences with conditioning prefix.
Conditioning tokens are derived from vocab_config.json produced by scan_dataset.py.
All bucket boundaries are data-driven — no hardcoded ranges.

Token layout:
  [0 .. COND_END)       — conditioning tokens (auto-sized)
  COND_END + 0          — PAD
  COND_END + 1          — BOS
  COND_END + 2          — EOS
  COND_END + 3          — BAR
  COND_END + 4..19      — POS_0..POS_15 (16th note positions)
  COND_END + 20..28     — TRACK_0..TRACK_8
  COND_END + 29..116    — PITCH_21..PITCH_108
  COND_END + 117..132   — DUR_1..DUR_16
  COND_END + 133..140   — VEL_1..VEL_8
  COND_END + 141..157   — TEMPO_40..TEMPO_200 step 10
  COND_END + 158        — SECTION_EARLY   (0-33% of song)
  COND_END + 159        — SECTION_MID     (33-66%)
  COND_END + 160        — SECTION_LATE    (66-100%)
"""

import json
from pathlib import Path

import numpy as np
import pretty_midi

# --------------------------------------------------------------------------- #
#  Load vocab config (set by init())
# --------------------------------------------------------------------------- #

_vocab_config: dict = {}
_bucket_config: dict = {}
COND_END: int = 0   # set by init()
VOCAB_SIZE: int = 0  # set by init()

# Fixed offsets relative to COND_END
_FIXED_BASE = 0
PAD:            int = 0
BOS:            int = 0
EOS:            int = 0
BAR:            int = 0
POS_OFFSET:     int = 0
TRACK_OFFSET:   int = 0
PITCH_OFFSET:   int = 0
DUR_OFFSET:     int = 0
VEL_OFFSET:     int = 0
TEMPO_OFFSET:   int = 0
SECTION_EARLY:  int = 0
SECTION_MID:    int = 0
SECTION_LATE:   int = 0

TEMPO_TOKENS = list(range(40, 201, 10))  # 17 values


def init(vocab_config_path: str | Path):
    """Must be called before encode/decode. Loads vocab config and sets all globals."""
    global _vocab_config, _bucket_config, COND_END, VOCAB_SIZE
    global PAD, BOS, EOS, BAR, POS_OFFSET, TRACK_OFFSET
    global PITCH_OFFSET, DUR_OFFSET, VEL_OFFSET, TEMPO_OFFSET
    global SECTION_EARLY, SECTION_MID, SECTION_LATE

    with open(vocab_config_path) as f:
        _vocab_config = json.load(f)
    _bucket_config = _vocab_config["bucket_config"]
    COND_END = _vocab_config["total_cond_tokens"]

    # Fixed tokens
    PAD           = COND_END + 0
    BOS           = COND_END + 1
    EOS           = COND_END + 2
    BAR           = COND_END + 3
    POS_OFFSET    = COND_END + 4    # +0..+15
    TRACK_OFFSET  = COND_END + 20   # +0..+8
    PITCH_OFFSET  = COND_END + 29   # +0..+87  (pitch 21..108)
    DUR_OFFSET    = COND_END + 117  # +0..+15  (dur 1..16)
    VEL_OFFSET    = COND_END + 133  # +0..+7   (vel 1..8)
    TEMPO_OFFSET  = COND_END + 141  # +0..+16  (17 tempos)
    SECTION_EARLY = COND_END + 158
    SECTION_MID   = COND_END + 159
    SECTION_LATE  = COND_END + 160

    VOCAB_SIZE = COND_END + 161


# --------------------------------------------------------------------------- #
#  Conditioning token encoding
# --------------------------------------------------------------------------- #

def _value_to_bucket(value: float, boundaries: list) -> int:
    for i, b in enumerate(boundaries):
        if value < b:
            return i
    return len(boundaries)


def _cat_to_bucket(value: int, values: list) -> int:
    try:
        return values.index(int(value))
    except ValueError:
        return 0


def encode_conditioning(features: dict) -> list[int]:
    """
    Encode a features dict (from scan_dataset.extract_features) into
    a list of conditioning token IDs, one per dimension.
    """
    tokens = []
    for field, cfg in _bucket_config.items():
        val = features.get(field, 0)
        offset = cfg["token_offset"]
        if cfg["type"] == "continuous":
            bucket = _value_to_bucket(float(val), cfg["boundaries"])
        else:
            bucket = _cat_to_bucket(val, cfg["values"])
        tokens.append(offset + bucket)
    return tokens


# --------------------------------------------------------------------------- #
#  MIDI helpers
# --------------------------------------------------------------------------- #

def _assign_slot(program: int, channel: int, is_drum: bool) -> int:
    if is_drum or channel == 9:
        return 4
    _SLOT_RULES = [
        (0,  set(range(0, 8))   | set(range(24, 32)) | set(range(56, 64)) | set(range(104, 112))),
        (1,  set(range(48, 56)) | set(range(88, 96))),
        (2,  set(range(24, 32)) | set(range(32, 40))),
        (3,  set(range(32, 40)) | {43}),
        (5,  set(range(40, 48))),
        (6,  set(range(56, 80))),
        (7,  set(range(16, 24))),
    ]
    for slot, programs in _SLOT_RULES:
        if program in programs:
            return slot
    return 8


def _bucket_velocity(vel: int) -> int:
    return max(1, min(8, int(vel / 16) + 1))


def _bucket_tempo(bpm: float) -> int:
    clamped = max(40, min(200, bpm))
    return round((clamped - 40) / 10)


def _section_token(bar_idx: int, total_bars: int) -> int:
    frac = bar_idx / max(total_bars, 1)
    if frac < 0.33:
        return SECTION_EARLY
    elif frac < 0.66:
        return SECTION_MID
    else:
        return SECTION_LATE


# --------------------------------------------------------------------------- #
#  Encode
# --------------------------------------------------------------------------- #

def encode(pm: pretty_midi.PrettyMIDI, features: dict | None = None) -> list[int]:
    """
    Encode a PrettyMIDI object to token ids.
    Prepends conditioning tokens if features dict is supplied.
    Injects SECTION_* tokens at each BAR for structural awareness.
    """
    # Build tempo map
    _tc_times, _tc_bpms = [], []
    for tick, spt in pm._tick_scales:
        t = pm.tick_to_time(tick)
        bpm = 60.0 / (spt * pm.resolution) if spt > 0 else 120.0
        _tc_times.append(t)
        _tc_bpms.append(bpm)

    def tempo_at(t: float) -> float:
        if not _tc_bpms:
            return 120.0
        idx = np.searchsorted(_tc_times, t, side="right") - 1
        return float(_tc_bpms[max(0, idx)])

    # Collect events
    events = []
    for instrument in pm.instruments:
        slot = _assign_slot(instrument.program, instrument.program, instrument.is_drum)
        for note in instrument.notes:
            bpm = tempo_at(note.start)
            sps = (60.0 / bpm) / 4  # seconds per 16th step
            raw_step = note.start / sps
            bar = int(raw_step // 16)
            pos = int(round(raw_step % 16)) % 16
            dur = max(1, min(16, round((note.end - note.start) / sps)))
            vel = _bucket_velocity(note.velocity)
            pitch = max(21, min(108, note.pitch))
            events.append((bar, pos, slot, pitch, dur, vel, bpm))

    if not events:
        return []

    max_bar = max(e[0] for e in events)
    bars: dict[int, list] = {b: [] for b in range(max_bar + 1)}
    for e in events:
        bars[e[0]].append(e)

    # Conditioning prefix
    tokens = []
    if features is not None:
        tokens.extend(encode_conditioning(features))

    tokens.append(BOS)
    last_tempo_idx = -1

    for bar_idx in range(max_bar + 1):
        bar_events = bars.get(bar_idx, [])
        if not bar_events:
            continue

        tokens.append(BAR)

        # Section token — tells model where it is in the piece
        tokens.append(_section_token(bar_idx, max_bar))

        # Tempo change token
        bar_bpm = bar_events[0][6]
        tempo_idx = _bucket_tempo(bar_bpm)
        if tempo_idx != last_tempo_idx:
            tokens.append(TEMPO_OFFSET + tempo_idx)
            last_tempo_idx = tempo_idx

        # Group by (pos, slot)
        pos_slot: dict[tuple, list] = {}
        for _, pos, slot, pitch, dur, vel, _ in bar_events:
            pos_slot.setdefault((pos, slot), []).append((pitch, dur, vel))

        for (pos, slot) in sorted(pos_slot.keys()):
            tokens.append(POS_OFFSET + pos)
            tokens.append(TRACK_OFFSET + slot)
            for pitch, dur, vel in sorted(pos_slot[(pos, slot)]):
                tokens.append(PITCH_OFFSET + (pitch - 21))
                tokens.append(DUR_OFFSET + (dur - 1))
                tokens.append(VEL_OFFSET + (vel - 1))

    tokens.append(EOS)
    return tokens


# --------------------------------------------------------------------------- #
#  Decode
# --------------------------------------------------------------------------- #

def decode(ids: list[int]) -> pretty_midi.PrettyMIDI:
    """Decode token ids to a PrettyMIDI object. Ignores conditioning prefix."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instruments = {}

    def get_instrument(slot: int) -> pretty_midi.Instrument:
        if slot not in instruments:
            is_drum = (slot == 4)
            programs = [0, 48, 25, 33, 0, 40, 56, 16, 0]
            inst = pretty_midi.Instrument(
                program=programs[slot] if not is_drum else 0,
                is_drum=is_drum,
                name=f"track_{slot}",
            )
            instruments[slot] = inst
            pm.instruments.append(inst)
        return instruments[slot]

    # Skip to BOS
    i = 0
    n = len(ids)
    while i < n and ids[i] != BOS:
        i += 1
    i += 1

    notes_raw = []
    bar_idx = 0
    bar_time = 0.0
    cur_tempo = 120.0
    bar_times: dict[int, float] = {}
    bar_tempos: dict[int, float] = {}

    while i < n:
        tok = ids[i]
        if tok == EOS:
            break
        elif tok == BAR:
            bar_idx += 1
            bar_times[bar_idx] = bar_time
            bar_tempos[bar_idx] = cur_tempo
            i += 1
            # Skip section token if present
            if i < n and ids[i] in (SECTION_EARLY, SECTION_MID, SECTION_LATE):
                i += 1
            # Tempo token
            if i < n and TEMPO_OFFSET <= ids[i] < TEMPO_OFFSET + len(TEMPO_TOKENS):
                cur_tempo = TEMPO_TOKENS[ids[i] - TEMPO_OFFSET]
                bar_tempos[bar_idx] = cur_tempo
                i += 1
            sps = (60.0 / cur_tempo) / 4
            bar_time += 16 * sps
        elif POS_OFFSET <= tok < POS_OFFSET + 16:
            cur_pos = tok - POS_OFFSET
            i += 1
            if i < n and TRACK_OFFSET <= ids[i] < TRACK_OFFSET + 9:
                cur_slot = ids[i] - TRACK_OFFSET
                i += 1
                while i < n and PITCH_OFFSET <= ids[i] < PITCH_OFFSET + 88:
                    pitch = ids[i] - PITCH_OFFSET + 21
                    i += 1
                    dur = 1
                    if i < n and DUR_OFFSET <= ids[i] < DUR_OFFSET + 16:
                        dur = ids[i] - DUR_OFFSET + 1
                        i += 1
                    vel = 4
                    if i < n and VEL_OFFSET <= ids[i] < VEL_OFFSET + 8:
                        vel = ids[i] - VEL_OFFSET + 1
                        i += 1
                    notes_raw.append((bar_idx, cur_pos, cur_slot, pitch, dur, vel))
        else:
            i += 1

    for bar, pos, slot, pitch, dur_steps, vel_bucket in notes_raw:
        b_time = bar_times.get(bar, 0.0)
        b_tempo = bar_tempos.get(bar, 120.0)
        sps = (60.0 / b_tempo) / 4
        t_start = b_time + pos * sps
        t_end = t_start + dur_steps * sps
        velocity = min(127, max(1, (vel_bucket - 1) * 16 + 8))
        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=t_start, end=t_end)
        get_instrument(slot).notes.append(note)

    for inst in pm.instruments:
        inst.notes.sort(key=lambda n: n.start)

    return pm


# --------------------------------------------------------------------------- #
#  Convenience: encode a file path
# --------------------------------------------------------------------------- #

def encode_file(path: str | Path, features: dict | None = None) -> list[int]:
    pm = pretty_midi.PrettyMIDI(str(path))
    return encode(pm, features)
