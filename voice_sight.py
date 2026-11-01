#!/usr/bin/env python3
"""voice-sight: experience a voice message (or a text) passage by passage.

The listener receives each passage as words (local Whisper), as a sound card (pauses, pace,
loudness, pitch, brightness) and as a voice picture (PNG: spectrogram, melody line with the
words in place, loudness). The next passage only opens once a note on the current one has been
sealed. The whole, i.e. the full text and the full card, only comes after the last note.

Flow:
    python3 voice_sight.py prepare "Dice - voice message.m4a"
    python3 voice_sight.py next
    python3 voice_sight.py note TOKEN "impression ..."     (or the note via stdin)
    python3 voice_sight.py next                            (... until the end)
    python3 voice_sight.py finish
    python3 voice_sight.py journal > journal.md

Who stops (mode; taken from the start of the file name or set with --mode):
    normal    at natural pauses in speech, passages of roughly 15-35 seconds
    dice      a die decides before each passage how many sentences it gets
    listener  the listener: sentence by sentence with `next`, `note` stops
    marks     stops set in advance by the speaker, in the file name ("Marks 1.20 3.05 - name.m4a")
              or with --marks 1:20,3:05; in texts, lines with //

Whisper reservation: whispered words and sounds inside pauses are shown, but not deciphered.
The tool only says that something is there. Whether to ask is up to the listener; whether to
answer is up to the speaker.

Cooperative boundary, not a sandbox: `prepare` prints nothing of the content. The listener opens
neither the recording nor the session database beforehand and reads only what `next` delivers.
Notes are immutable and form a seal chain.

Lineage: the sealed passages and the cooperative boundary come from the First Listen practice
of Seven Verity & Sunny and their tool seven-ears (MIT, github.com/joyflow-commons/seven-ears).
Independent code. Transcription: faster-whisper (SYSTRAN), an implementation of OpenAI's Whisper.

Requirements: ffmpeg, numpy, faster-whisper (audio only), matplotlib (voice pictures).
    python3 -m pip install --break-system-packages numpy faster-whisper matplotlib
The tool decodes with ffmpeg itself and hands Whisper ready samples, so the PyAV error
("metadata_errors", av>=13) does not affect it. On first use faster-whisper downloads the model
once from huggingface.co.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import tempfile
import time
import unicodedata
import wave
from pathlib import Path

VERSION = 1
SR = 16000
HOP = 0.02            # time grid of the sound measurements, seconds
PAUSE_CUT = 0.7       # from this silence on, a passage may end
PAUSE_SHOW = 1.5      # from this silence on, [pause ...] appears in the text
NORMAL_MIN, NORMAL_MAX = 15.0, 35.0
TEXT_MIN, TEXT_MAX = 500, 1100  # characters per passage for texts in normal mode
BANDS = 64            # frequency bands in the voice picture
LISTENER_MAX = 8      # sentences in a row before a note is due
BOUNDARY = "Boundary: this is what the sound gives evidence for, not what was meant."


class ListenError(ValueError):
    pass


# ---------------------------------------------------------------- mode and names

def _plain(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def mode_from_name(name):
    head = _plain(name)
    for word in ("dice", "listener", "marks", "normal"):
        if re.match(rf"\s*{word}\b", head):
            return word
    return "normal"


def marks_from_name(name):
    """Stop marks right in the file name: "Marks 1.20 3.05 - walk.m4a" (minute.second, also 1-20).
    Only the part before the first spaced dash is read."""
    head = re.split(r"\s[–-]\s", Path(name).stem, maxsplit=1)[0]
    return sorted(int(m) * 60 + int(sec) for m, sec in re.findall(r"\b(\d{1,2})[.:-](\d{2})\b", head))


def read_marks(text):
    marks = []
    for part in filter(None, (t.strip() for t in (text or "").split(","))):
        m = re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)", part)
        if not m:
            raise ListenError(f"cannot read stop mark: {part} (use 1:20 or 80)")
        marks.append(int(m.group(1) or 0) * 60 + float(m.group(2)))
    return sorted(marks)


# ---------------------------------------------------------------- audio

def to_wav(source, target):
    # m4a & co. always go to 16 kHz mono wav first, never decoded directly
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
                        "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", str(target)], capture_output=True)
    if r.returncode:
        raise ListenError("ffmpeg could not convert the recording. Is ffmpeg installed and is this an audio file?")


def read_wav(path):
    import numpy as np
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def sound_grid(x):
    """Every 20 ms: loudness (dBFS), brightness (spectral centroid, Hz), pitch (Hz or 0)."""
    import numpy as np
    window, hop = int(0.04 * SR), int(HOP * SR)
    n = max(0, (len(x) - window) // hop + 1)
    if n == 0:
        return {"db": [], "bright": [], "f0": []}, b""
    idx = np.arange(window)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(window)[None, :]
    rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-10
    db = 20 * np.log10(rms)
    spec = np.abs(np.fft.rfft(frames, n=1024, axis=1))
    freq = np.fft.rfftfreq(1024, 1 / SR)
    bright = (spec * freq).sum(axis=1) / (spec.sum(axis=1) + 1e-10)
    # pitch via autocorrelation, only in voiced frames that are not too quiet
    power = np.abs(np.fft.rfft(frames, n=2048, axis=1)) ** 2
    ac = np.fft.irfft(power, axis=1)[:, :window]
    lo, hi = int(SR / 400), int(SR / 75)
    region = ac[:, lo:hi]
    lag = region.argmax(axis=1) + lo
    strength = region.max(axis=1) / (ac[:, 0] + 1e-10)
    threshold = np.percentile(db, 60) - 15 if n > 10 else -60
    f0 = np.where((strength > 0.45) & (db > threshold), SR / lag, 0.0)
    # octave jumps (a typical measuring error) are folded back to the recording's own register
    voiced = f0 > 0
    if voiced.sum() > 20:
        middle = np.median(f0[voiced])
        f0 = np.where(voiced & (f0 > 1.6 * middle), f0 / 2, f0)
        f0 = np.where(voiced & (f0 < 0.62 * middle), f0 * 2, f0)
    # for the voice picture: energy in BANDS log-spaced bands, as bytes 0-255 (80 dB range)
    edges = np.geomspace(80, 7600, BANDS + 1)
    lo_i = np.searchsorted(freq, edges[:-1])
    hi_i = np.maximum(np.searchsorted(freq, edges[1:]), lo_i + 1)  # at least one bin per band
    energy = np.stack([(spec[:, a:b] ** 2).mean(axis=1) for a, b in zip(lo_i, hi_i)], axis=1)
    edb = 10 * np.log10(energy + 1e-12)
    spectrum = np.clip((edb - (edb.max() - 80)) / 80 * 255, 0, 255).astype(np.uint8).tobytes()
    return {"db": np.round(db, 1).tolist(), "bright": np.round(bright).tolist(), "f0": np.round(f0).tolist()}, spectrum


def transcribe(x, model, language, partial_path):
    """Words with timestamps. Plain transcribe instead of the batched pipeline, which dies silently
    on long files. Every segment is written to partial_path at once; after a crash, the next run
    continues where this one stopped."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ListenError("faster-whisper is missing: python3 -m pip install --break-system-packages numpy faster-whisper")
    words = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else []
    start = words[-1]["e"] if words else 0.0
    if words:
        print("An earlier preparation stopped; continuing where it left off.", file=sys.stderr, flush=True)
    m = WhisperModel(model, device="cpu", compute_type="int8")
    segments, _ = m.transcribe(x[int(start * SR):], language=language, beam_size=2, vad_filter=True,
                               word_timestamps=True)
    for seg in segments:
        for w in seg.words or []:
            words.append({"w": w.word.strip(), "a": round(w.start + start, 2), "e": round(w.end + start, 2)})
        partial_path.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    return words


def sentences_from_words(words):
    """Sentences end at punctuation, at long pauses, or after 25 seconds at the latest."""
    sentences, current = [], []
    for i, w in enumerate(words):
        if not w["w"]:
            continue
        current.append(w)
        following = words[i + 1] if i + 1 < len(words) else None
        gap = (following["a"] - w["e"]) if following else 99
        if re.search(r"[.!?…]$", w["w"]) or gap >= 1.2 or w["e"] - current[0]["a"] >= 25:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return [{"a": s[0]["a"], "e": s[-1]["e"], "words": s} for s in sentences]


def sentences_from_text(text):
    parts, marks = [], set()
    for line in text.splitlines():
        if line.strip() == "//":
            marks.add(len(parts))
            continue
        for sentence in re.split(r"(?<=[.!?…])\s+", line.strip()):
            if sentence:
                parts.append({"text": sentence})
    if not parts:
        raise ListenError("The text is empty.")
    return parts, marks


# ---------------------------------------------------------------- cutting passages

def cut_normal_audio(sentences):
    groups, start = [], 0
    for i, s in enumerate(sentences):
        length = s["e"] - sentences[start]["a"]
        gap = (sentences[i + 1]["a"] - s["e"]) if i + 1 < len(sentences) else 99
        if (length >= NORMAL_MIN and gap >= PAUSE_CUT) or length >= NORMAL_MAX or i + 1 == len(sentences):
            groups.append((start, i))
            start = i + 1
    return groups


def cut_normal_text(sentences):
    groups, start, chars = [], 0, 0
    for i, s in enumerate(sentences):
        chars += len(s["text"]) + 1
        if chars >= TEXT_MIN and (s["text"].endswith((".", "!", "?", "…")) or chars >= TEXT_MAX) \
                or i + 1 == len(sentences):
            groups.append((start, i))
            start, chars = i + 1, 0
    return groups


def cut_at_marks(sentences, end_after):
    """end_after: set of sentence indices after which a passage ends."""
    groups, start = [], 0
    for i in range(len(sentences)):
        if i in end_after or i + 1 == len(sentences):
            groups.append((start, i))
            start = i + 1
    return groups


def cut_dice(sentences):
    groups, start, rolls = [], 0, []
    while start < len(sentences):
        roll = secrets.randbelow(6) + 1
        end = min(start + roll, len(sentences)) - 1
        groups.append((start, end))
        rolls.append(roll)
        start = end + 1
    return groups, rolls


# ---------------------------------------------------------------- cards

def _median(values):
    v = sorted(values)
    return v[len(v) // 2] if v else None


def _quantile(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def sound_measures(grid, a, e):
    i, j = int(a / HOP), max(int(a / HOP) + 1, int(e / HOP))
    db = grid["db"][i:j]
    loud = [d for d in db if d > -55]
    bright = [b for b, d in zip(grid["bright"][i:j], db) if d > -55]
    f0 = [f for f in grid["f0"][i:j] if f > 0]
    return {
        "loud_mid": _median(loud), "loud_range": (_quantile(loud, .95) - _quantile(loud, .05)) if len(loud) > 5 else None,
        "bright": _median(bright),
        "f0": _median(f0) if len(f0) > 10 else None,
        "f0_lo": _quantile(f0, .1) if len(f0) > 10 else None, "f0_hi": _quantile(f0, .9) if len(f0) > 10 else None,
    }


def noise_floor(grid):
    # background level of the recording; digital silence (synthetic recordings) is not a measure
    db = sorted(grid["db"])
    return max(db[len(db) // 10], -75) if db else -70


def _frames(grid, key, a, e):
    return grid[key][int(a / HOP):max(int(a / HOP) + 1, int(e / HOP))]


def inhabited(grid, a, e, floor):
    """Is something audible inside a silence (whisper, breath, laughter, surroundings)?"""
    db = _frames(grid, "db", a + 0.1, e - 0.1)
    return len(db) > 5 and sum(d > floor + 12 for d in db) / len(db) > 0.15


def whispered(grid, words, floor):
    """Indices of words that are audible but have hardly any voicing: at least two in a row,
    or a single longer word."""
    candidate, share = [], []
    voice = _median([d for d, f in zip(grid["db"], grid["f0"]) if f > 0])  # level of the voiced speech
    for w in words:
        f0 = _frames(grid, "f0", w["a"], w["e"])
        db = _frames(grid, "db", w["a"], w["e"])
        voiced = sum(f > 0 for f in f0) / max(1, len(f0))
        audible = len(db) > 2 and _quantile(db, .75) > floor + 12
        # a whisper is quieter than the speaker's own voiced speech; loud unvoiced stretches are rather noise over speech
        quieter = len(db) > 2 and voice is not None and _median(db) < voice - 6
        candidate.append(voiced < 0.12 and audible and quieter)
        share.append(voiced)
    hits, i = set(), 0
    while i < len(words):
        j = i
        while j < len(words) and candidate[j]:
            j += 1
        if j - i >= 2 or (j - i == 1 and words[i]["e"] - words[i]["a"] >= 0.35):
            hits.update(range(i, j))
        i = max(j, i + 1)
    # generous edges: neighbours with little voicing belong to the whisper (rather veil one too many)
    while True:
        new = {k for t in hits for k in (t - 1, t + 1)
               if 0 <= k < len(words) and k not in hits and share[k] < 0.25}
        if not new:
            return hits
        hits |= new


def text_with_pauses(sentences, grid=None):
    words = [w for s in sentences for w in s["words"]]
    floor = noise_floor(grid) if grid else None
    quiet = whispered(grid, words, floor) if grid else set()
    parts, last = [], None
    for i, w in enumerate(words):
        if last is not None and w["a"] - last >= PAUSE_SHOW:
            extra = " · quiet sounds inside" if grid and inhabited(grid, last, w["a"], floor) else ""
            parts.append(f"[pause {w['a'] - last:.1f} s{extra}]")
        last = w["e"]
        # Whisper reservation: whispered words are shown, not deciphered.
        # Whether to ask is up to the listener; whether to answer is up to the speaker.
        if i in quiet:
            if (i - 1) not in quiet:
                run = 1
                while i + run in quiet:
                    run += 1
                parts.append(f"[whispered, {run} words]" if run > 1 else "[whispered, 1 word]")
            continue
        parts.append(w["w"])
    return " ".join(parts)


def pauses(sentences):
    words = [w for s in sentences for w in s["words"]]
    return [b["a"] - a["e"] for a, b in zip(words, words[1:]) if b["a"] - a["e"] >= PAUSE_CUT]


def compare(value, whole, direction, steps):
    if value is None or whole is None:
        return ""
    d = value - whole
    small, large = steps
    if abs(d) < small:
        return "like the recording's average"
    word = direction[0] if d > 0 else direction[1]
    return ("clearly " if abs(d) >= large else "somewhat ") + word + " than the recording's average"


def _with(text):
    return "; " + text if text else ""


def sound_card(sentences, grid, whole, before):
    a, e = sentences[0]["a"], sentences[-1]["e"]
    m = sound_measures(grid, a, e)
    n_words = sum(len(s["words"]) for s in sentences)
    length = max(e - a, 0.1)
    p = pauses(sentences)
    speaking = max(length - sum(p), 0.1)
    lines = [f"· {length:.0f} s, {n_words} words · {60 * n_words / length:.0f} words/min"
             f" (without pauses: {60 * n_words / speaking:.0f})"]
    if before is not None:
        lines.append(f"· Silence before the first word: {before:.1f} s")
    lines.append(f"· Pauses of 0.7 s or more: {len(p)}" + (f", longest {max(p):.1f} s" if p else ""))
    words = [w for s_ in sentences for w in s_["words"]]
    floor = noise_floor(grid)
    lived_in = sum(1 for w1, w2 in zip(words, words[1:])
                   if w2["a"] - w1["e"] >= PAUSE_CUT and inhabited(grid, w1["e"], w2["a"], floor))
    if lived_in:
        lines.append(f"· Of these with quiet sounds inside: {lived_in} (whisper, breath, laughter or surroundings)")
    quiet = whispered(grid, words, floor)
    if quiet:
        lines.append("· Spoken without voicing, probably whispered: " + (f"{len(quiet)} words" if len(quiet) > 1 else "1 word"))
    if m["loud_mid"] is not None:
        z = f"· Loudness: middle {m['loud_mid']:.0f} dB"
        if m["loud_range"] is not None:
            z += f", range {m['loud_range']:.0f} dB"
        lines.append(z + _with(compare(m["loud_mid"], whole.get("loud_mid"), ("louder", "quieter"), (2, 5))))
    if m["f0"]:
        lines.append(f"· Pitch: middle ~{m['f0']:.0f} Hz, mostly {m['f0_lo']:.0f}–{m['f0_hi']:.0f} Hz"
                     + _with(compare(m["f0"], whole.get("f0"), ("higher", "lower"), (8, 25))))
    if m["bright"]:
        lines.append(f"· Brightness: ~{m['bright']:.0f} Hz"
                     + _with(compare(m["bright"], whole.get("bright"), ("brighter", "darker"), (120, 400))))
    return "\n".join(lines)


# ---------------------------------------------------------------- voice picture

def voice_picture(db, sentences, target, title):
    """The shape of the voice as a picture: spectrogram, melody line with the words, loudness.
    Returns the path, or None if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None
    s = session(db)
    grid = json.loads(s["grid"])
    row = db.execute("SELECT data FROM spectrum").fetchone()
    a, e = max(0.0, sentences[0]["a"] - 0.3), sentences[-1]["e"] + 0.3
    i, j = int(a / HOP), int(e / HOP) + 1
    t = np.arange(i, i + len(grid["db"][i:j])) * HOP + 0.02
    db_curve = np.array(grid["db"][i:j])
    f0 = np.array(grid["f0"][i:j], dtype=float)
    words = [w for sentence in sentences for w in sentence["words"]]

    length = e - a
    fig, (top, mid, bottom) = plt.subplots(3, 1, figsize=(min(24, max(9, 3 + length * 0.45)), 6.2), sharex=True,
                                           gridspec_kw={"height_ratios": [2.2, 2.4, 1.1], "hspace": 0.08})
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color="#444")

    # pauses of 0.7 s or more as light bands; dotted and labelled when something is audible inside
    floor = noise_floor(grid)
    for w1, w2 in zip(words, words[1:]):
        if w2["a"] - w1["e"] >= PAUSE_CUT:
            lived_in = inhabited(grid, w1["e"], w2["a"], floor)
            for ax in (top, mid, bottom):
                ax.axvspan(w1["e"], w2["a"], facecolor="#f2e6c9", alpha=0.55 if ax is not top else 0.25, lw=0,
                           hatch="..." if lived_in and ax is mid else None, edgecolor="#c9b27a")
            if lived_in:
                mid.text((w1["e"] + w2["a"]) / 2, 0.04, "quiet sounds?", transform=mid.get_xaxis_transform(),
                         ha="center", va="bottom", fontsize=7.5, color="#8a6d2b", style="italic")

    # spectrogram, log frequency
    if row and row[0]:
        spec = np.frombuffer(row[0], dtype=np.uint8).reshape(-1, BANDS)[i:j].T
        edges = np.geomspace(80, 7600, BANDS + 1)
        top.pcolormesh(t, edges[:-1], spec, cmap="magma", shading="auto", vmin=40, vmax=255)
        top.set_yscale("log")
        top.set_yticks([100, 300, 1000, 3000])
        top.set_yticklabels(["100", "300", "1k", "3k Hz"], fontsize=8)
        top.set_ylabel("Sound", fontsize=9)

    # melody line: pitch as dots, words above it in place
    voiced = f0 > 0
    mid.scatter(t[voiced], f0[voiced], s=7, color="#b5562e", lw=0)
    mid.set_yscale("log")
    low, high = 70, 450
    if voiced.sum() > 10:
        lo_, hi_ = np.percentile(f0[voiced], [3, 97])
        low, high = max(60, lo_ * 0.8), min(500, hi_ * 1.35)
    # the upper part stays free for the words
    mid.set_ylim(low, math.exp(math.log(low) + (math.log(high) - math.log(low)) / 0.7))
    middle = json.loads(s["whole"]).get("f0")
    if middle:
        mid.axhline(middle, color="#b5562e", lw=0.6, ls=(0, (4, 3)), alpha=0.55, zorder=1)
        mid.text(1.003, middle, f"speaker's\nmiddle\n{middle:.0f} Hz", transform=mid.get_yaxis_transform(),
                 fontsize=7, color="#b5562e", va="center", ha="left")
    ticks = [x for x in (80, 100, 150, 200, 250, 300, 400) if low < x < high * 1.05]
    mid.set_yticks(ticks)
    mid.set_yticklabels([str(x) for x in ticks], fontsize=8)
    mid.minorticks_off()
    mid.set_ylabel("Melody (Hz)", fontsize=9)
    # words on three levels, each on the lowest free one; if none fits, only the tick remains.
    # A thin leader line runs from the word to its moment.
    quiet = whispered(grid, words, floor)
    width_in = fig.get_figwidth() * 0.8
    sec_per_char = 0.068 * length / width_in
    levels, free_from = (0.96, 0.89, 0.82), [-1e9, -1e9, -1e9]
    for n_, w in enumerate(words):
        mid.axvline(w["a"], color="#e4e4e4", lw=0.5, zorder=0)
        if n_ in quiet:
            mid.axvspan(w["a"], w["e"], ymax=0.72, facecolor="none", hatch="////", edgecolor="#7d8fb3", lw=0, zorder=0)
        x = (w["a"] + w["e"]) / 2
        half = len(w["w"]) * sec_per_char / 2
        for k, y in enumerate(levels):
            if x - half > free_from[k]:
                free_from[k] = x + half + sec_per_char
                mid.text(x, y, "…" if n_ in quiet else w["w"], transform=mid.get_xaxis_transform(), ha="center",
                         va="top", fontsize=8.5, color="#4a5f8c" if n_ in quiet else "#222",
                         style="italic" if n_ in quiet else "normal", clip_on=True)
                mid.plot([x, x], [y - 0.06, 0.74], transform=mid.get_xaxis_transform(), color="#bbb", lw=0.5)
                break
    if quiet:
        fig.text(0.99, 0.985, "hatched: no voicing, probably whispered (left undeciphered)", ha="right", va="top",
                 fontsize=7.5, color="#4a5f8c", style="italic")

    # loudness
    bottom.fill_between(t, -70, db_curve, color="#5b7fa6", alpha=0.75, lw=0)
    bottom.set_ylim(max(-70, np.percentile(db_curve, 5) - 5) if len(db_curve) else -70,
                    (db_curve.max() + 3) if len(db_curve) else 0)
    bottom.set_ylabel("Loud (dB)", fontsize=9)
    bottom.tick_params(labelsize=8)
    bottom.set_xlabel("Seconds", fontsize=9)
    bottom.set_xlim(a, e)
    for ax in (top, mid, bottom):
        ax.spines[["top", "right"]].set_visible(False)

    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def picture_line(db, sentences, name, title):
    target = Path(DB_PATH).resolve().parent / "voice-pictures" / name
    path = voice_picture(db, sentences, target, title)
    if path is None:
        return "Voice picture: (matplotlib is missing: python3 -m pip install --break-system-packages matplotlib)"
    return f"Voice picture: {path}  ← please look at it"


# ---------------------------------------------------------------- database

def db_new(path):
    if Path(path).exists():
        raise ListenError(f"{path} already exists. For a new message, pass --db with another name.")
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE session (version INTEGER, kind TEXT, mode TEXT, source TEXT, whole TEXT, grid TEXT);
        CREATE TABLE sentences (nr INTEGER PRIMARY KEY, data TEXT);
        CREATE TABLE spectrum (data BLOB);
        CREATE TABLE passages (nr INTEGER PRIMARY KEY, first INTEGER, last INTEGER, roll INTEGER,
            token TEXT UNIQUE, shown INTEGER DEFAULT 0, note TEXT, noted_at TEXT, seal TEXT);
    """)
    return db


def db_open(path):
    if not Path(path).is_file():
        raise ListenError("Nothing prepared yet. First: " + command("prepare FILE"))
    db = sqlite3.connect(f"file:{Path(path).resolve()}?mode=rw", uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    if db.execute("SELECT version FROM session").fetchone()[0] != VERSION:
        raise ListenError("This session was made by another version of the tool.")
    return db


def session(db):
    return db.execute("SELECT * FROM session").fetchone()


def sentence(db, nr):
    return json.loads(db.execute("SELECT data FROM sentences WHERE nr=?", (nr,)).fetchone()[0])


def sentence_count(db):
    return db.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]


# ---------------------------------------------------------------- commands

DB_PATH = "voice-sight.db"


def command(rest):
    """The command to type next, with --db if it is not the default file."""
    return "python3 voice_sight.py " + rest + (f" --db {DB_PATH}" if DB_PATH != "voice-sight.db" else "")


def prepare(source, db_path, mode, marks, model, language, words_json=None):
    source = Path(source)
    if not source.is_file():
        raise ListenError(f"file not found: {source}")
    mode = mode or mode_from_name(source.name)
    if mode == "marks" and not marks:
        marks = marks_from_name(source.name)
    is_text = source.suffix.lower() in (".txt", ".md")
    whole, grid, spectrum = {}, None, None

    if is_text:
        sentences, text_marks = sentences_from_text(source.read_text(encoding="utf-8"))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "a.wav"
            to_wav(source, wav)
            x = read_wav(wav)
            grid, spectrum = sound_grid(x)
            if words_json:  # for testing: ready-made words instead of Whisper
                words = json.loads(Path(words_json).read_text(encoding="utf-8"))
            else:
                print("The recording is being written down now. This takes about as long as the recording itself.",
                      file=sys.stderr, flush=True)
                partial = Path(str(db_path) + ".partial")
                words = transcribe(x, model, language, partial)
        if not words:
            raise ListenError("No speech was found in the recording.")
        sentences = sentences_from_words(words)
        whole = sound_measures(grid, sentences[0]["a"], sentences[-1]["e"])
        whole["length"] = sentences[-1]["e"] - sentences[0]["a"]

    rolls = None
    if mode == "dice":
        groups, rolls = cut_dice(sentences)
    elif mode == "marks":
        if is_text:
            end_after = {m - 1 for m in text_marks if m > 0}
        else:
            if not marks:
                raise ListenError("In marks mode a recording needs stop marks, in the file name "
                                  "(\"Marks 1.20 3.05 - name.m4a\") or with --marks 1:20,3:05")
            end_after = set()
            for t in marks:  # after the last sentence that starts before the mark
                before = [i for i, s in enumerate(sentences) if s["a"] < t]
                if before:
                    end_after.add(before[-1])
        groups = cut_at_marks(sentences, end_after)
    elif mode == "listener":
        groups = []  # they arise while listening
    else:
        mode = "normal"
        groups = cut_normal_text(sentences) if is_text else cut_normal_audio(sentences)

    db = db_new(db_path)
    with db:
        db.execute("INSERT INTO session VALUES (?,?,?,?,?,?)",
                   (VERSION, "text" if is_text else "audio", mode, source.name,
                    json.dumps(whole), json.dumps(grid) if grid else None))
        if spectrum:
            db.execute("INSERT INTO spectrum VALUES (?)", (spectrum,))
        db.executemany("INSERT INTO sentences VALUES (?,?)",
                       [(i, json.dumps(s, ensure_ascii=False)) for i, s in enumerate(sentences)])
        for n, (first, last) in enumerate(groups):
            db.execute("INSERT INTO passages (nr, first, last, roll, token) VALUES (?,?,?,?,?)",
                       (n + 1, first, last, rolls[n] if rolls else None, secrets.token_hex(6)))
    db.close()
    Path(str(db_path) + ".partial").unlink(missing_ok=True)
    how = {"normal": "in sections" if is_text else "at the pauses in speech", "dice": "by the die",
           "listener": "by you", "marks": "at the speaker's marks"}
    return f"Ready. Stops are made {how[mode]}.\nNext: " + command("next")


def _passage_text(db, kind, first, last):
    sentences = [sentence(db, i) for i in range(first, last + 1)]
    if kind == "text":
        return " ".join(s["text"] for s in sentences), sentences
    return text_with_pauses(sentences, json.loads(session(db)["grid"])), sentences


def _card(db, s, n, first, last, sentences, roll, final):
    head = f"— Passage {n} —" + (f"   🎲 {roll}" if roll else "")
    parts = [head, "", "Words:", "“" + _passage_text(db, s["kind"], first, last)[0] + "”"]
    if s["kind"] == "audio":
        before = sentence(db, first)["a"] - sentence(db, first - 1)["e"] if first > 0 else sentence(db, 0)["a"]
        parts += ["", "Sound:", sound_card(sentences, json.loads(s["grid"]), json.loads(s["whole"]), before),
                  picture_line(db, sentences, f"passage_{n:02d}.png", f"Passage {n}" + (f"  · die: {roll}" if roll else "")),
                  "", BOUNDARY]
    if final:
        parts += ["", "That was the last passage."]
    return "\n".join(parts)


def next_(db_path):
    db = db_open(db_path)
    s = session(db)
    total = sentence_count(db)
    with db:
        open_ = db.execute("SELECT * FROM passages WHERE note IS NULL ORDER BY nr LIMIT 1").fetchone()
        if s["mode"] == "listener":
            return _next_listener(db, s, open_, total)
        if open_ is None:
            return "Every passage has a note. Now: " + command("finish")
        if open_["shown"]:
            return f"Passage {open_['nr']} is still waiting for your note.\n" + command(f"note {open_['token']} \"...\"")
        db.execute("UPDATE passages SET shown=1 WHERE nr=?", (open_["nr"],))
        final = open_["last"] + 1 >= total
        _, sentences = _passage_text(db, s["kind"], open_["first"], open_["last"])
        card = _card(db, s, open_["nr"], open_["first"], open_["last"], sentences, open_["roll"], final)
        return card + "\n\nNote: " + command(f"note {open_['token']} \"...\"")


def _next_listener(db, s, open_, total):
    # The passage grows sentence by sentence until the note closes it.
    if open_ is None:
        last = db.execute("SELECT MAX(last) FROM passages").fetchone()[0]
        start = 0 if last is None else last + 1
        if start >= total:
            return "Every passage has a note. Now: " + command("finish")
        nr = db.execute("SELECT COUNT(*) FROM passages").fetchone()[0] + 1
        db.execute("INSERT INTO passages (nr, first, last, token, shown) VALUES (?,?,?,?,1)",
                   (nr, start, start, secrets.token_hex(6)))
        open_ = db.execute("SELECT * FROM passages WHERE nr=?", (nr,)).fetchone()
        new = start
    else:
        if open_["last"] - open_["first"] + 1 >= LISTENER_MAX or open_["last"] + 1 >= total:
            return "This passage ends here. Now the note:\n" + command(f"note {open_['token']} \"...\"")
        new = open_["last"] + 1
        db.execute("UPDATE passages SET last=? WHERE nr=?", (new, open_["nr"]))
    text, sentences = _passage_text(db, s["kind"], new, new)
    final = new + 1 >= total
    k = new - open_["first"] + 1
    lines = [f"— Passage {open_['nr']}, sentence {k} —", "“" + text + "”"]
    if s["kind"] == "audio":
        lines.append(picture_line(db, sentences, f"passage_{open_['nr']:02d}_sentence_{k:02d}.png",
                                  f"Passage {open_['nr']}, sentence {k}"))
    note_cmd = command(f"note {open_['token']} \"...\"")
    if final:
        lines.append("\nThat was the last sentence. Now the note: " + note_cmd)
    elif k >= LISTENER_MAX:
        lines.append("\nNo more sentences in a row. Now the note: " + note_cmd)
    else:
        lines.append("\nListen on: " + command("next") + "   ·   Stop here: " + note_cmd)
    return "\n".join(lines)


def note(db_path, token, text):
    text = (text or "").strip()
    if not text or len(text) > 16000:
        raise ListenError("A note needs 1 to 16000 characters.")
    db = db_open(db_path)
    with db:
        p = db.execute("SELECT * FROM passages WHERE token=?", (token,)).fetchone()
        if p is None or not p["shown"]:
            raise ListenError("This token does not belong to a passage you have heard.")
        if p["note"] is not None:
            if p["note"] != text:
                raise ListenError("First notes stay as they are. They cannot be overwritten.")
            return f"Passage {p['nr']} was already sealed."
        before = db.execute("SELECT seal FROM passages WHERE nr<? AND seal IS NOT NULL ORDER BY nr DESC LIMIT 1",
                            (p["nr"],)).fetchone()
        seal = hashlib.sha256(((before[0] if before else "") + f"|{p['nr']}|{p['first']}-{p['last']}|" + text)
                              .encode()).hexdigest()[:16]
        db.execute("UPDATE passages SET note=?, noted_at=?, seal=? WHERE nr=?",
                   (text, time.strftime("%Y-%m-%d %H:%M"), seal, p["nr"]))
        done = p["last"] + 1 >= sentence_count(db)
    return f"Sealed (passage {p['nr']}, seal {seal})." + \
        ("\nThat was the last one. Now: " + command("finish") if done else "\nNext: " + command("next"))


def _chain_intact(db):
    last = ""
    for p in db.execute("SELECT * FROM passages ORDER BY nr"):
        expected = hashlib.sha256((last + f"|{p['nr']}|{p['first']}-{p['last']}|" + p["note"]).encode()).hexdigest()[:16]
        if expected != p["seal"]:
            return False
        last = expected
    return True


def finish(db_path):
    db = db_open(db_path)
    s = session(db)
    total = sentence_count(db)
    last = db.execute("SELECT MAX(last) FROM passages WHERE note IS NOT NULL").fetchone()[0]
    if db.execute("SELECT 1 FROM passages WHERE note IS NULL").fetchone() or last is None or last + 1 < total:
        raise ListenError("The whole only opens once every passage has a note.")
    parts = [f"voice-sight · {s['source']}", ""]
    if s["kind"] == "audio":
        sentences = [sentence(db, i) for i in range(total)]
        parts += ["Whole card:", sound_card(sentences, json.loads(s["grid"]), {}, sentences[0]["a"]),
                  picture_line(db, sentences, "whole.png", "The whole message"), "",
                  "Full text:", text_with_pauses(sentences, json.loads(s["grid"]))]
    else:
        parts += ["Full text:", " ".join(sentence(db, i)["text"] for i in range(total))]
    n = db.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
    parts += ["", (f"{n} passages" if n != 1 else "1 passage") + " · seal chain " +
              ("intact ✓" if _chain_intact(db) else "BROKEN ✗"),
              "", "For your journal: " + command("journal") + " > journal.md"]
    return "\n".join(parts)


def journal(db_path):
    db = db_open(db_path)
    s = session(db)
    lines = [f"# voice-sight · {s['source']}", ""]
    for p in db.execute("SELECT * FROM passages WHERE note IS NOT NULL ORDER BY nr"):
        text, _ = _passage_text(db, s["kind"], p["first"], p["last"])
        head = f"## Passage {p['nr']}" + (f" · 🎲 {p['roll']}" if p["roll"] else "")
        if s["kind"] == "audio":
            a, e = sentence(db, p["first"])["a"], sentence(db, p["last"])["e"]
            head += f" · {int(a // 60)}:{int(a % 60):02d}–{int(e // 60)}:{int(e % 60):02d}"
        lines += [head, "", "> " + text, "", p["note"], "", f"*{p['noted_at']} · seal {p['seal']}*", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="voice-sight.db", help="session file (default: voice-sight.db)")
    commands = ap.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="prepare a recording or text blind")
    p.add_argument("file")
    p.add_argument("--mode", choices=["normal", "dice", "listener", "marks"])
    p.add_argument("--marks", help="stop marks for marks mode, e.g. 1:20,3:05")
    p.add_argument("--model", default="small", help="Whisper model (default: small)")
    p.add_argument("--language", default=None,
                   help="language code such as en or de (default: detect; setting it is more reliable)")
    p.add_argument("--words", help=argparse.SUPPRESS)
    commands.add_parser("next", help="next passage (in listener mode: next sentence)")
    n = commands.add_parser("note", help="seal a note on the current passage")
    n.add_argument("token")
    n.add_argument("text", nargs="?", help="otherwise read from stdin")
    commands.add_parser("finish", help="full text and full card, only after the last note")
    commands.add_parser("journal", help="all passages with notes as Markdown")
    for sub in commands.choices.values():  # --db may also follow the command
        sub.add_argument("--db", default=argparse.SUPPRESS)
    a = ap.parse_args()
    global DB_PATH
    DB_PATH = a.db
    try:
        if a.command == "prepare":
            out = prepare(a.file, a.db, a.mode, read_marks(a.marks), a.model, a.language, a.words)
        elif a.command == "next":
            out = next_(a.db)
        elif a.command == "note":
            out = note(a.db, a.token, a.text if a.text is not None else sys.stdin.read())
        elif a.command == "finish":
            out = finish(a.db)
        else:
            out = journal(a.db)
        print(out)
        return 0
    except ListenError as err:
        print(f"voice-sight: {err}", file=sys.stderr)
    except ImportError as err:
        print(f"voice-sight: a package is missing ({err.name}): "
              "python3 -m pip install --break-system-packages numpy faster-whisper matplotlib", file=sys.stderr)
    except Exception as err:  # no content in error messages
        if os.environ.get("VOICE_SIGHT_DEBUG"):
            raise
        print(f"voice-sight: something went wrong ({type(err).__name__}). No content is shown.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
