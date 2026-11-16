"""Does the voice get duller the longer a sentence runs? High-band energy and spectral centroid over time.

    python muffle.py <served dir> <report.md>      # tts image (librosa); <dir>/manifest.json from served_ab.py
    python muffle.py tail <served dir> <report.md>  # the tail set: the same closing clause, early vs late
For each clip, per 1-s window of speech frames (10 ms frames within 35 dB of the clip's loudest): the share of
energy above 4 kHz (dB) and the spectral centroid (Hz), both at 24 kHz. Trend = last 4 s minus first 4 s, and a
linear fit in dB per 10 s. The training recordings of similar length are measured the same way: that is the target.
"""
import glob, json, sys, wave
import numpy as np

SR, HOP, NFFT = 24000, 240, 1024
D = "/yuke/raisa-voice-data600/wavs/"


def series(path):
    import librosa
    y, _ = librosa.load(path, sr=SR)
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    total = S.sum(0) + 1e-12
    db = 10 * np.log10(total)
    speech = db > db.max() - 35
    hf = 10 * np.log10(S[freqs >= 4000].sum(0) / total + 1e-12)
    centroid = (freqs[:, None] * S).sum(0) / total
    t = np.arange(S.shape[1]) * HOP / SR
    on = t[speech]
    wins = []
    for start in np.arange(on[0], on[-1], 1.0):
        sel = speech & (t >= start) & (t < start + 1)
        if sel.sum() >= 20:
            wins.append((float(start - on[0]), float(np.mean(hf[sel])), float(np.mean(centroid[sel]))))
    return {"dur": float(on[-1] - on[0]), "wins": wins}


def trend(s):
    w = np.array(s["wins"])
    if len(w) < 6:
        return None
    first, last = w[w[:, 0] < 4], w[w[:, 0] >= w[-1, 0] - 3]
    return {"dur": s["dur"], "hf_first": first[:, 1].mean(), "hf_drop": last[:, 1].mean() - first[:, 1].mean(),
            "cen_first": first[:, 2].mean(), "cen_ratio": last[:, 2].mean() / first[:, 2].mean(),
            "slope": np.polyfit(w[:, 0], w[:, 1], 1)[0] * 10, "curve": w[:, 1].tolist()}


def main(out, path):
    man = json.load(open(out + "/manifest.json"))
    gen = [(f, trend(series(f"{out}/{f['file']}"))) for f in man["files"] if f["set"] == "length"]
    real = []
    for p in sorted(glob.glob(D + "*.wav")):
        w = wave.open(p)
        if w.getnframes() / w.getframerate() >= 8:
            t = trend(series(p))
            if t:
                real.append(t)
    L = ["# Does the voice get duller over a long sentence?", "",
         "Per clip: high band = energy above 4 kHz as a share of all (dB), averaged over the first 4 s of speech; drop = last 4 s",
         "minus first 4 s (dB, negative = duller at the end); slope = linear fit over all 1-s windows (dB per 10 s); centroid =",
         "spectral centroid, last 4 s / first 4 s. Median over the clips in each row.", "",
         "| clips | n | speech s | high band first 4 s dB | drop dB | slope dB/10 s | centroid first Hz | centroid last/first |",
         "|---|---|---|---|---|---|---|---|"]
    def row(name, ts):
        ts = [t for t in ts if t]
        if not ts:
            return
        m = lambda k: float(np.median([t[k] for t in ts]))
        L.append(f"| {name} | {len(ts)} | {m('dur'):.1f} | {m('hf_first'):.1f} | {m('hf_drop'):+.1f} | {m('slope'):+.2f} | {m('cen_first'):.0f} | {m('cen_ratio'):.2f} |")
    for lo, hi in ((8, 12), (12, 18), (18, 40)):
        row(f"training recordings {lo}-{hi} s", [t for t in real if lo <= t["dur"] < hi])
    for secs in sorted({f["target_s"] for f, _ in gen}):
        for lang in ("id", "en"):
            row(f"served, ~{secs} s, {lang}", [t for f, t in gen if f["target_s"] == secs and f["lang"] == lang])
    L += ["", "High band per 1-s window (dB), served clips of the two longest lengths, median over the 3 samples:", ""]
    for secs in sorted({f["target_s"] for f, _ in gen})[-2:]:
        for lang in ("id", "en"):
            curves = [t["curve"] for f, t in gen if t and f["target_s"] == secs and f["lang"] == lang]
            n = min(map(len, curves))
            L.append(f"- ~{secs} s {lang}: " + " ".join(f"{np.median([c[i] for c in curves]):.1f}" for i in range(n)))
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def tail_stats(path, bounds=(), rate=48000, secs=2.5):
    """The last `secs` of speech: high band (dB) over all speech frames and over voiced frames, centroid, level;
    fade = level of the last `secs` minus the first; pause = silence around each request boundary (s)."""
    import librosa
    y, _ = librosa.load(path, sr=SR)
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    total = S.sum(0) + 1e-12
    db = 10 * np.log10(total)
    speech = np.flatnonzero(db > db.max() - 35)
    t = np.arange(S.shape[1]) * HOP / SR
    sel = speech[t[speech] >= t[speech[-1]] - secs]
    _, voiced, _ = librosa.pyin(y, fmin=70, fmax=600, sr=SR, hop_length=HOP, frame_length=NFFT)
    v = sel[voiced[sel[sel < len(voiced)]]] if len(voiced) else sel
    first = speech[t[speech] <= t[speech[0]] + secs]
    quiet = db <= db.max() - 35
    pauses = []
    for b in bounds:
        i = min(int(b / rate * SR / HOP), len(quiet) - 1)
        lo = hi = i
        while lo > 0 and quiet[lo - 1]:
            lo -= 1
        while hi < len(quiet) and quiet[hi]:
            hi += 1
        pauses.append((hi - lo) * HOP / SR)
    hf = lambda idx: float(10 * np.log10(S[freqs >= 4000][:, idx].sum() / total[idx].sum()))
    return {"hf": hf(sel), "hf_voiced": hf(v), "centroid": float((freqs[:, None] * S[:, sel]).sum() / total[sel].sum()),
            "level": float(10 * np.log10(total[sel].mean())), "dur": float(t[speech[-1]] - t[speech[0]]),
            "fade": float(10 * np.log10(total[sel].mean() / total[first].mean())), "pauses": pauses}


def tail(out, path):
    man = json.load(open(out + "/manifest.json"))
    rows = [(f, tail_stats(f"{out}/{f['file']}", f["bounds"][:-1], f["rate"])) for f in man["files"] if f["set"] == "tail"]
    L = ["# The same closing clause, alone vs at the end of a long sentence", "",
         "Last 2.5 s of speech of each clip (the same words in every row of a language): high band = energy above 4 kHz as a",
         "share of all (dB), over all speech frames and over voiced frames only; centroid; level (dB, relative). Median of 6",
         "samples, then the change from the clause said alone. Negative high band / centroid change = duller. Fade = level of",
         "the last 2.5 s minus the first 2.5 s of the same clip (training recordings: -1.1 dB median). split = the sentence sent",
         "in tts_pieces; pause = silence where one piece ends and the next starts; levelled = tts/server.Leveller (level_sim.py).", "",
         "| lang | clauses before | sent as | sentence s | high band dB | voiced high band dB | centroid Hz | level dB | fade dB | change vs alone: high band / voiced / centroid / level | pause at cut s |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lang in ("id", "en"):
        base = None
        for n, variant in [(n, v) for n in (0, 1, 3, 5) for v in ("served", "split", "levelled")]:
            st = [x for f, x in rows if f["lang"] == lang and f["clauses"] == n and f["variant"] == variant]
            if not st:
                continue
            m = {k: float(np.median([x[k] for x in st])) for k in st[0] if k != "pauses"}
            pauses = [p for x in st for p in x["pauses"]]
            base = base or m
            L.append(f"| {lang} | {n} | {dict(served='one request', split='pieces', levelled='one request, levelled')[variant]} | {m['dur']:.1f} | {m['hf']:.1f} | {m['hf_voiced']:.1f} | "
                     f"{m['centroid']:.0f} | {m['level']:.1f} | {m['fade']:+.1f} | {m['hf'] - base['hf']:+.1f} / {m['hf_voiced'] - base['hf_voiced']:+.1f} / "
                     f"{m['centroid'] / base['centroid'] - 1:+.0%} / {m['level'] - base['level']:+.1f} | "
                     + (f"{np.median(pauses):.2f} ({min(pauses):.2f}-{max(pauses):.2f})" if pauses else "") + " |")
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    tail(*sys.argv[2:]) if sys.argv[1] == "tail" else main(*sys.argv[1:])
