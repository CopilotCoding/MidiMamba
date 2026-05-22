"""
tokenizer.py — MidiGen3

MIDI ↔ token encoding.  Ported from midigen2 with two bug fixes:

  Fix 1: decode() bar_times/bar_tempos now initialized at bar 0 before the
          loop, so the first bar's tempo is always correct (was silently using
          120 BPM fallback if first bar had a different tempo).

  Fix 2: Added decode_to_file() convenience wrapper.

Token layout (identical to midigen2 — checkpoints are compatible):
  [0 .. COND_END)       — conditioning tokens (data-driven, from vocab_config.json)
  COND_END + 0          — PAD
  COND_END + 1          — BOS
  COND_END + 2          — EOS
  COND_END + 3          — BAR
  COND_END + 4..19      — POS_0..POS_15  (16th note positions within bar)
  COND_END + 20..28     — TRACK_0..TRACK_8
  COND_END + 29..116    — PITCH_21..PITCH_108
  COND_END + 117..132   — DUR_1..DUR_16
  COND_END + 133..140   — VEL_1..VEL_8
  COND_END + 141..157   — TEMPO_40..TEMPO_200 step 10
  COND_END + 158        — SECTION_EARLY  (0–33% of song)
  COND_END + 159        — SECTION_MID    (33–66%)
  COND_END + 160        — SECTION_LATE   (66–100%)
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi

# --------------------------------------------------------------------------- #
#  Module-level state (set by init())
# --------------------------------------------------------------------------- #

_vocab_config:  dict = {}
_bucket_config: dict = {}
COND_END:   int = 0
VOCAB_SIZE: int = 0

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

TEMPO_TOKENS = list(range(40, 201, 10))   # 17 values: 40, 50, ..., 200


def init(vocab_config_path):
    """Must be called before encode/decode.  Loads vocab config, sets all globals."""
    global _vocab_config, _bucket_config, COND_END, VOCAB_SIZE
    global PAD, BOS, EOS, BAR, POS_OFFSET, TRACK_OFFSET
    global PITCH_OFFSET, DUR_OFFSET, VEL_OFFSET, TEMPO_OFFSET
    global SECTION_EARLY, SECTION_MID, SECTION_LATE

    with open(vocab_config_path) as f:
        _vocab_config = json.load(f)
    _bucket_config = _vocab_config["bucket_config"]
    COND_END = _vocab_config["total_cond_tokens"]

    PAD           = COND_END + 0
    BOS           = COND_END + 1
    EOS           = COND_END + 2
    BAR           = COND_END + 3
    POS_OFFSET    = COND_END + 4
    TRACK_OFFSET  = COND_END + 20
    PITCH_OFFSET  = COND_END + 29
    DUR_OFFSET    = COND_END + 117
    VEL_OFFSET    = COND_END + 133
    TEMPO_OFFSET  = COND_END + 141
    SECTION_EARLY = COND_END + 158
    SECTION_MID   = COND_END + 159
    SECTION_LATE  = COND_END + 160

    VOCAB_SIZE = COND_END + 161


# --------------------------------------------------------------------------- #
#  Conditioning
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


def encode_conditioning(features: dict) -> list:
    tokens = []
    for field, cfg in _bucket_config.items():
        val    = features.get(field, 0)
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
        (0, set(range(0,  8))  | set(range(24, 32)) | set(range(56, 64)) | set(range(104, 112))),
        (1, set(range(48, 56)) | set(range(88, 96))),
        (2, set(range(24, 32)) | set(range(32, 40))),
        (3, set(range(32, 40)) | {43}),
        (5, set(range(40, 48))),
        (6, set(range(56, 80))),
        (7, set(range(16, 24))),
    ]
    for slot, programs in _SLOT_RULES:
        if program in programs:
            return slot
    return 8


def _bucket_velocity(vel: int) -> int:
    return max(1, min(8, int(vel / 16) + 1))


def _bucket_tempo(bpm: float) -> int:
    return round((max(40, min(200, bpm)) - 40) / 10)


def _section_token(bar_idx: int, total_bars: int) -> int:
    frac = bar_idx / max(total_bars, 1)
    if frac < 0.33:  return SECTION_EARLY
    elif frac < 0.66: return SECTION_MID
    else:             return SECTION_LATE


# --------------------------------------------------------------------------- #
#  Encode
# --------------------------------------------------------------------------- #

def encode(pm: pretty_midi.PrettyMIDI, features: Optional[dict] = None) -> list:
    """Encode PrettyMIDI → token id list.  Prepends conditioning if features given."""
    _tc_times, _tc_bpms = [], []
    for tick, spt in pm._tick_scales:
        t   = pm.tick_to_time(tick)
        bpm = 60.0 / (spt * pm.resolution) if spt > 0 else 120.0
        _tc_times.append(t)
        _tc_bpms.append(bpm)

    def tempo_at(t: float) -> float:
        if not _tc_bpms: return 120.0
        idx = np.searchsorted(_tc_times, t, side="right") - 1
        return float(_tc_bpms[max(0, idx)])

    events = []
    for instrument in pm.instruments:
        slot = _assign_slot(instrument.program, instrument.program, instrument.is_drum)
        for note in instrument.notes:
            bpm     = tempo_at(note.start)
            sps     = (60.0 / bpm) / 4
            raw     = note.start / sps
            bar     = int(raw // 16)
            pos     = int(round(raw % 16)) % 16
            dur     = max(1, min(16, round((note.end - note.start) / sps)))
            vel     = _bucket_velocity(note.velocity)
            pitch   = max(21, min(108, note.pitch))
            events.append((bar, pos, slot, pitch, dur, vel, bpm))

    if not events:
        return []

    max_bar = max(e[0] for e in events)
    bars: dict = {b: [] for b in range(max_bar + 1)}
    for e in events:
        bars[e[0]].append(e)

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
        tokens.append(_section_token(bar_idx, max_bar))

        bar_bpm    = bar_events[0][6]
        tempo_idx  = _bucket_tempo(bar_bpm)
        if tempo_idx != last_tempo_idx:
            tokens.append(TEMPO_OFFSET + tempo_idx)
            last_tempo_idx = tempo_idx

        pos_slot: dict = {}
        for _, pos, slot, pitch, dur, vel, _ in bar_events:
            pos_slot.setdefault((pos, slot), []).append((pitch, dur, vel))

        for (pos, slot) in sorted(pos_slot.keys()):
            tokens.append(POS_OFFSET + pos)
            tokens.append(TRACK_OFFSET + slot)
            for pitch, dur, vel in sorted(pos_slot[(pos, slot)]):
                tokens.append(PITCH_OFFSET + (pitch - 21))
                tokens.append(DUR_OFFSET   + (dur  - 1))
                tokens.append(VEL_OFFSET   + (vel  - 1))

    tokens.append(EOS)
    return tokens


def encode_file(path, features: Optional[dict] = None) -> list:
    return encode(pretty_midi.PrettyMIDI(str(path)), features)


# --------------------------------------------------------------------------- #
#  Decode  (BUG FIX: bar 0 initialized correctly before loop)
# --------------------------------------------------------------------------- #

def decode(ids: list) -> pretty_midi.PrettyMIDI:
    """Decode token id list → PrettyMIDI.  Conditioning prefix is ignored."""
    pm          = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instruments = {}

    def get_instrument(slot: int) -> pretty_midi.Instrument:
        if slot not in instruments:
            programs   = [0, 48, 25, 33, 0, 40, 56, 16, 0]
            is_drum    = (slot == 4)
            inst       = pretty_midi.Instrument(
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

    notes_raw  = []
    bar_idx    = 0
    cur_tempo  = 120.0

    # FIX: initialize bar 0 before loop so it's always present
    bar_times:  dict = {0: 0.0}
    bar_tempos: dict = {0: cur_tempo}
    bar_time    = 0.0

    while i < n:
        tok = ids[i]
        if tok == EOS:
            break
        elif tok == BAR:
            bar_idx += 1
            bar_times[bar_idx]  = bar_time
            bar_tempos[bar_idx] = cur_tempo
            i += 1
            if i < n and ids[i] in (SECTION_EARLY, SECTION_MID, SECTION_LATE):
                i += 1
            if i < n and TEMPO_OFFSET <= ids[i] < TEMPO_OFFSET + len(TEMPO_TOKENS):
                cur_tempo = TEMPO_TOKENS[ids[i] - TEMPO_OFFSET]
                bar_tempos[bar_idx] = cur_tempo
                i += 1
            sps      = (60.0 / cur_tempo) / 4
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
        b_time  = bar_times.get(bar,  0.0)
        b_tempo = bar_tempos.get(bar, 120.0)
        sps     = (60.0 / b_tempo) / 4
        t_start = b_time + pos * sps
        t_end   = t_start + dur_steps * sps
        velocity = min(127, max(1, (vel_bucket - 1) * 16 + 8))
        get_instrument(slot).notes.append(
            pretty_midi.Note(velocity=velocity, pitch=pitch, start=t_start, end=t_end)
        )

    for inst in pm.instruments:
        inst.notes.sort(key=lambda n: n.start)

    return pm


def decode_to_file(ids: list, path) -> None:
    """Decode token ids and write a .mid file directly."""
    decode(ids).write(str(path))
