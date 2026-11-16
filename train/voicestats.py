"""Shared speech measures: trimmed duration, f0 median, onset pitch ratio (first 0.4 s vs rest)."""
import numpy as np, librosa
HOP = 160
def measure(path):
    y, sr = librosa.load(path, sr=16000)
    yt, _ = librosa.effects.trim(y, top_db=35)
    f0, vf, _ = librosa.pyin(yt, fmin=70, fmax=600, sr=sr, hop_length=HOP)
    v = f0[vf]
    head, tail = f0[:40][vf[:40]], f0[40:][vf[40:]]
    onset = float(np.median(head) / np.median(tail)) if len(head) >= 5 and len(tail) >= 10 else float("nan")
    return {"dur": len(yt) / sr, "f0": float(np.median(v)) if len(v) else float("nan"), "onset": onset}
