"""Offline A/B of tts/server.Leveller: replay served renders through it in the server's chunk size, then measure.

    python level_sim.py <served dir with the tail set> <out dir>     # tts image, repo at /rv
Writes <out>/tail/*.wav (the same clips, levelled) and a manifest, so muffle.py tail can compare.
"""
import json, os, sys, wave
import numpy as np

sys.path.insert(0, "/rv/tts")
from server import Leveller  # noqa: E402

CHUNK = 7680  # one generation step at 48 kHz


def main(src, out):
    man = json.load(open(src + "/manifest.json"))
    files = [f for f in man["files"] if f["set"] == "tail" and f["variant"] == "served"]
    os.makedirs(out + "/tail", exist_ok=True)
    for f in files:
        with wave.open(f"{src}/{f['file']}") as w:
            rate, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
        level = Leveller(rate)
        y = np.concatenate([level(x[i:i + CHUNK]) for i in range(0, len(x), CHUNK)])
        with wave.open(f"{out}/{f['file']}", "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
            w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    json.dump({**man, "files": [{**f, "variant": "levelled"} for f in files]}, open(out + "/manifest.json", "w"))
    # self-check: a steady tone keeps its level, a tone that fades 6 dB over 10 s ends within 3 dB of its start (ALLOW_DB + the average lags)
    t = np.arange(rate * 10) / rate
    tone = 0.3 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    for fade_db in (0, 6):
        lv = Leveller(rate)
        x = tone * 10 ** (-fade_db * t / 10 / 20)
        y = np.concatenate([lv(x[i:i + CHUNK]) for i in range(0, len(x), CHUNK)])
        end = 20 * np.log10(np.sqrt(np.mean(y[-rate:] ** 2)) / np.sqrt(np.mean(y[:rate] ** 2)))
        assert (abs(end) < 0.1) if fade_db == 0 else (-3.0 < end < -1.0), (fade_db, end)
    print("levelled", len(files))


if __name__ == "__main__":
    main(*sys.argv[1:])
