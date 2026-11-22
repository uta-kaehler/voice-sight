"""Whisper reservation: words recognised as whispered must never be written to disk.

Synthetic only: a generated recording (a tone for the voice, quiet noise for the whisper, near-silence around it)
and a stand-in for faster-whisper that hands out fixed words. No real recordings, no Whisper model.

Run: python3 tests/test_whisper_boundary.py
"""
import json
import math
import random
import sqlite3
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import voice_sight as vs  # noqa: E402

SECRET = ["Geheimnis", "Flüsterwort", "verborgen"]
# (word, start, end, kind) — "v" voiced, "w" whispered
SCRIPT = [
    ("We", 0.50, 0.80, "v"), ("arrived", 0.85, 1.40, "v"), ("at", 1.45, 1.65, "v"), ("the", 1.70, 1.90, "v"),
    ("gate", 1.95, 2.60, "v"),
    (SECRET[0], 3.00, 3.60, "w"), (SECRET[1], 3.65, 4.30, "w"), (SECRET[2], 4.35, 5.00, "w"),
    ("See", 5.40, 5.70, "v"), ("you", 5.75, 6.00, "v"), ("tomorrow.", 6.05, 6.90, "v"),
]
LENGTH = 7.6


def make_recording(path):
    rnd = random.Random(7)
    n = int(LENGTH * vs.SR)
    x = [rnd.uniform(-1, 1) * 0.0004 for _ in range(n)]          # near-silence
    for _, a, e, kind in SCRIPT:
        for i in range(int(a * vs.SR), int(e * vs.SR)):
            t = i / vs.SR
            if kind == "v":                                        # voice: 140 Hz with harmonics
                x[i] += 0.30 * sum(math.sin(2 * math.pi * 140 * h * t) / h for h in (1, 2, 3, 4))
            else:                                                  # whisper: quiet noise, no pitch
                x[i] += rnd.uniform(-1, 1) * 0.05
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(vs.SR)
        w.writeframes(b"".join(int(max(-1, min(1, v)) * 32767).to_bytes(2, "little", signed=True) for v in x))


class _Word:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = " " + word, start, end


class _Segment:
    def __init__(self, words):
        self.words = words


def fake_whisper(segment_ends, crash_after=None):
    """A stand-in for faster_whisper: fixed words, cut into segments; optionally crashes after a segment."""
    class WhisperModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **k):
            offset = LENGTH - len(audio) / vs.SR                   # transcribe() passes x[start:]
            def segments():
                bounds = [0.0] + segment_ends
                for n, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
                    ws = [_Word(w, a - offset, e - offset) for w, a, e, _ in SCRIPT if lo <= a < hi and a >= offset - 0.01]
                    if ws:
                        yield _Segment(ws)
                    if crash_after is not None and n == crash_after:
                        raise KeyboardInterrupt("simulated crash")
            return segments(), None
    return types.SimpleNamespace(WhisperModel=WhisperModel)


class WhisperBoundary(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.audio = self.tmp / "Normal - test.wav"
        make_recording(self.audio)
        self.db = self.tmp / "session.db"
        self.writes = []
        original = Path.write_text

        def spy(path, data, *a, **k):                              # record every write, to any file
            self.writes.append((str(path), data))
            return original(path, data, *a, **k)
        Path.write_text = spy
        self.addCleanup(setattr, Path, "write_text", original)

    def assertNoSecret(self, text, where):
        for s in SECRET:
            self.assertNotIn(s, text, f"whispered word found in {where}")

    def test_detection_works_on_the_synthetic_recording(self):
        x = vs.read_wav(self.audio)
        grid, _ = vs.sound_grid(x)
        words = [{"w": w, "a": a, "e": e} for w, a, e, _ in SCRIPT]
        hits = vs.whispered(grid, words, vs.noise_floor(grid))
        self.assertEqual({5, 6, 7}, hits & {5, 6, 7}, "the three whispered words are recognised")
        self.assertFalse(hits & {0, 1, 2, 9, 10}, "clearly voiced words stay readable")

    def test_crash_and_resume_never_write_the_whisper(self):
        # segment 1 ends in the middle of the whisper, then the run crashes
        sys.modules["faster_whisper"] = fake_whisper([3.62, 99], crash_after=0)
        with self.assertRaises(KeyboardInterrupt):
            vs.prepare(self.audio, self.db, None, None, "small", "en")
        partial = Path(str(self.db) + ".partial")
        self.assertTrue(partial.exists(), "the crash leaves a .partial file behind")
        self.assertNoSecret(partial.read_text(encoding="utf-8"), ".partial after the crash")
        # resume, this time to the end
        sys.modules["faster_whisper"] = fake_whisper([3.62, 99])
        vs.prepare(self.audio, self.db, None, None, "small", "en")
        self.assertFalse(partial.exists(), ".partial is removed after a finished preparation")
        self.assertTrue(self.writes, "the spy saw writes")
        for path, data in self.writes:
            self.assertNoSecret(data, f"a write to {Path(path).name}")
        self.assertNoSecret(self.db.read_bytes().decode("utf-8", "replace"), "the session file")

    def test_output_shows_the_whisper_without_its_words(self):
        sys.modules["faster_whisper"] = fake_whisper([99])
        vs.prepare(self.audio, self.db, None, None, "small", "en")
        out = vs.next_(self.db)
        self.assertNoSecret(out, "the passage shown")
        self.assertIn("[whispered, 3 words]", out)
        self.assertIn("arrived", out)
        con = sqlite3.connect(self.db)
        stored = [json.loads(r[0]) for r in con.execute("SELECT * FROM sentences").fetchall() for r in [r[1:]]]
        con.close()
        veiled = [w for s in stored for w in s["words"] if w.get("q")]
        self.assertEqual(3, len(veiled), "position and count of the whisper remain")
        self.assertTrue(all(w["w"] == vs.VEIL for w in veiled))


if __name__ == "__main__":
    unittest.main(verbosity=2)
