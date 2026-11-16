"""Score smooth_gen.py output: sentence joins, drift, tempo, Whisper CER, short sentences, colok-words.

    python smooth_ana.py measure <dir>        # tts image on the compose network (librosa, phones.py, asr service)
    python smooth_ana.py report <dir> <report.md> [<served dir>]   # served_ab.py renders join as "+trim" rows
Join = where one sentence ends and the next starts. For each join: the silence across it, and the step in
pitch (semitones) and level (dB) from the last 0.4 s of speech before it to the first 0.4 s after it.
The training recordings with several sentences are measured the same way: that is the target, not zero.
"""
import io, json, os, re, sys, urllib.request, wave
import numpy as np

HOP, SR, WIN = 160, 16000, 40  # 10 ms frames; 0.4 s windows
D = "/yuke/raisa-voice-data600/"
ASR = os.environ.get("ASR_URL", "http://asr:8000")


def syllables(text: str) -> int:
    return max(1, len(re.findall(r"[aiueo]+", text.lower())))


def frames(path):
    import librosa
    y, _ = librosa.load(path, sr=SR)
    f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=600, sr=SR, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=4 * HOP, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    f0 = np.where(voiced[:n], f0[:n], np.nan)
    active = rms[:n] > max(rms.max() * 0.03, 1e-4)  # about -30 dB under the peak
    return y, f0, rms[:n], active


def silences(active):
    """Inner runs of silence as (start, end) frames, longest first."""
    runs, start = [], None
    for i, a in enumerate(active):
        if not a and start is None:
            start = i
        elif a and start is not None:
            runs.append((start, i)); start = None
    on = np.flatnonzero(active)
    runs = [r for r in runs if len(on) and r[0] > on[0] and r[1] <= on[-1]]
    return sorted(runs, key=lambda r: r[0] - r[1])


def med(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else float("nan")


def measure_reply(path, n_sents, bounds_frames=None):
    """Joins and per-sentence figures; joins are the given boundaries, else the n-1 longest silences."""
    y, f0, rms, active = frames(path)
    on = np.flatnonzero(active)
    if not len(on):
        return None
    if bounds_frames is None:
        joins = sorted((a + b) // 2 for a, b in silences(active)[: n_sents - 1])
    else:
        joins = bounds_frames
    edges = [int(on[0])] + list(joins) + [int(on[-1]) + 1]
    sents, out_joins = [], []
    for a, b in zip(edges, edges[1:]):
        idx = np.arange(a, b)[active[a:b]]
        if not len(idx):
            sents.append(None); continue
        seg_f0 = f0[idx[0]: idx[-1] + 1]
        head, tail = seg_f0[:WIN], seg_f0[WIN:]
        ok = np.isfinite(head).sum() >= 5 and np.isfinite(tail).sum() >= 10
        sents.append({"start": int(idx[0]), "end": int(idx[-1]) + 1, "f0": med(seg_f0), "active_s": len(idx) * HOP / SR,
                      "onset": med(head) / med(tail) if ok else float("nan"),
                      "level": float(20 * np.log10(np.sqrt(np.mean(rms[idx] ** 2)) + 1e-9)),
                      "f0_first": med(f0[idx[:WIN]]), "f0_last": med(f0[idx[-WIN:]]),
                      "lv_first": float(20 * np.log10(np.mean(rms[idx[:WIN]]) + 1e-9)),
                      "lv_last": float(20 * np.log10(np.mean(rms[idx[-WIN:]]) + 1e-9))})
    for a, b in zip(sents, sents[1:]):
        if a and b:
            out_joins.append({"gap_s": (b["start"] - a["end"]) * HOP / SR, "f0_step": float(12 * np.log2(b["f0_first"] / a["f0_last"])),
                              "level_step": b["lv_first"] - a["lv_last"]})
    return {"lead_s": int(on[0]) * HOP / SR, "dur_s": len(y) / SR, "sents": sents, "joins": out_joins}


def whisper(path, lang_hint=""):
    import librosa
    y, _ = librosa.load(path, sr=SR)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    req = urllib.request.Request(f"{ASR}/v1/transcribe?hotwords=halo", data=buf.getvalue(), method="POST")
    return json.load(urllib.request.urlopen(req, timeout=60))["text"]


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def cer(hyp, ref):
    norm = lambda t: " ".join(re.sub(r"[^\w\s]", " ", t.lower()).split())
    return lev(norm(hyp), norm(ref)) / max(1, len(norm(ref)))


def measure(out):
    from acro import g2p, heard, region
    from phones import Phones
    man = json.load(open(out + "/manifest.json"))
    rate, phones = man["rate"], Phones()
    man["files"] = [m for m in man["files"] if m["set"] in ("replies", "short", "lexicon")]  # muffle.py / gemini take the rest
    for m in man["files"]:
        path = f"{out}/{m['file']}"
        bounds = [int(b * SR / rate / HOP) for b in m["bounds"][:-1]] if len(m["bounds"]) > 1 else None
        m["m"] = measure_reply(path, len(m["sents"]), bounds)
        m["heard"] = whisper(path)
        if m["set"] == "lexicon":
            left, right = m["sents"][0].split(m["word"])
            m["phones"] = "".join(region(heard(phones(path)), g2p(left), g2p(m["word"]), g2p(right)))
    # The training recordings with two or more sentences: what the voice was taught a join is.
    real = []
    for line in open(D + "voxcpm2_lora_600_dataset.jsonl"):
        row = json.loads(line)
        n = len(re.findall(r"[.!?](?=\s+\S)", row["text"])) + 1
        if 2 <= n <= 4:
            r = measure_reply(D + row["audio"], n)
            if r and len(r["joins"]) == n - 1:
                real.append({"text": row["text"], "m": r})
    man["real"] = real
    json.dump(man, open(out + "/measured.json", "w"), ensure_ascii=False, indent=0)
    print("measured", len(man["files"]), "files,", len(real), "real clips")


def report(out, path, served=None):
    man = json.load(open(out + "/measured.json"))
    files = man["files"]
    if served:  # rendered through the live tts after the lead-silence trim
        more = json.load(open(served + "/measured.json"))
        files += [{**f, "variant": f["variant"] + "+trim"} for f in more["files"] if f["set"] in ("replies", "short")]
        man["solo_ms"]["served"] = more["solo_ms"]["fixed"]
    q = lambda xs, p: float(np.nanpercentile(xs, p)) if len(xs) and np.isfinite(xs).any() else float("nan")

    def row(name, ms, texts=None, extra=""):
        joins = [j for m in ms for j in m["joins"]]
        sents = [[s for s in m["sents"] if s] for m in ms]
        spread = lambda key: med([np.std([s[key] for s in ss]) for ss in sents if len(ss) > 1])
        f0_sd = med([np.std([12 * np.log2(s["f0"]) for s in ss]) for ss in sents if len(ss) > 1])
        drift = [12 * np.log2(ss[-1]["f0"] / ss[0]["f0"]) for ss in sents if len(ss) > 1]
        onsets = np.array([s["onset"] for ss in sents for s in ss], dtype=float)
        rates = [np.std([syllables(t) / s["active_s"] for t, s in zip(tx, ss)]) for tx, ss in zip(texts or [], sents) if len(tx) == len(ss) > 1]
        g = [j["gap_s"] for j in joins]
        return (f"| {name} | {len(joins)} | {med(g):.2f} ({q(g, 10):.2f}-{q(g, 90):.2f}) | {med([j['f0_step'] for j in joins]):+.1f} | "
                f"{med([abs(j['f0_step']) for j in joins]):.1f} | {med([abs(j['level_step']) for j in joins]):.1f} | {f0_sd:.2f} | {spread('level'):.1f} | "
                f"{med(rates):.2f} | {med(drift):+.1f} ({q(np.abs(drift), 90):.1f}) | {med([s['f0'] for ss in sents for s in ss]):.0f} | "
                f"{np.nanmedian(onsets):.2f} / {int((onsets > 1.3).sum())} of {int(np.isfinite(onsets).sum())} | {med([m['lead_s'] for m in ms]):.2f} |{extra}")

    L = ["# Sentence continuity (smooth_gen.py)", "",
         "Joins: silence across the join (median, 10-90%), pitch step in semitones (signed median, then median size), level step (dB).",
         "Within an answer: spread (sd) of sentence pitch (semitones), level (dB) and tempo (syllables/s); drift = pitch of the last",
         "sentence minus the first (median, then the 90% size). Onset = first 0.4 s pitch / rest, per sentence. Lead = silence before the first word.", "",
         "| variant | joins | gap s | pitch step | size | level step | pitch sd | level sd | tempo sd | drift st | f0 Hz | onset / >1.3x | lead s | CER | first chunk ms: k=1 / k>1 | encode ms |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    L.append(row("training audio (2-4 sentences)", [r["m"] for r in man["real"]]) + " | | |")
    for v in ("fixed", "whole", "cont", "conttrim", "refcont", "served+trim"):
        fs = [f for f in files if f["set"] == "replies" and f["variant"] == v and f["m"]]
        if not fs:
            continue
        c = np.mean([cer(f["heard"], " ".join(f["sents"])) for f in fs])
        k1 = med([f["first_ms"][0] for f in fs]); kn = med([x for f in fs for x in f["first_ms"][1:]])
        enc = med([x for f in fs for x in f["encode_ms"]]) if any(f["encode_ms"] for f in fs) else float("nan")
        L.append(row(v, [f["m"] for f in fs], None if v == "whole" else [f["sents"] for f in fs],
                     f" {c:.3f} | {k1:.0f} / {kn:.0f} | {enc:.0f} |"))
    s = man["solo_ms"]
    L += ["", f"First chunk with nothing else on the GPU (6 pairs): fixed prompt {med(s['fixed']):.0f} ms, continuation {med(s['cont']):.0f} ms "
          f"+ encoding the previous sentence {med(s['encode']):.0f} ms (nano-vllm in-process, not through HTTP)."]
    if "served" in s:
        L += [f"Served TTS first byte after the trim (HTTP, one at a time, {len(s['served'])} sentences): median {med(s['served']):.0f} ms "
              f"({q(s['served'], 10):.0f}-{q(s['served'], 90):.0f}); that byte is now speech, before it was about 0.5 s of silence."]
    L += ["",
          "## Per language", "", "| variant | lang | gap s | pitch step size | pitch sd | drift st | CER |", "|---|---|---|---|---|---|---|"]
    for v in ("fixed", "whole", "cont", "conttrim", "refcont", "served+trim"):
        for lang, pick in (("id", lambda f: f["reply"] < 10), ("en", lambda f: f["reply"] >= 10)):
            fs = [f for f in files if f["set"] == "replies" and f["variant"] == v and f["m"] and pick(f)]
            if not fs:
                continue
            joins = [j for f in fs for j in f["m"]["joins"]]
            sents = [[s for s in f["m"]["sents"] if s] for f in fs]
            L.append(f"| {v} | {lang} | {med([j['gap_s'] for j in joins]):.2f} | {med([abs(j['f0_step']) for j in joins]):.1f} | "
                     f"{med([np.std([12 * np.log2(s['f0']) for s in ss]) for ss in sents if len(ss) > 1]):.2f} | "
                     f"{med([12 * np.log2(ss[-1]['f0'] / ss[0]['f0']) for ss in sents if len(ss) > 1]):+.1f} | "
                     f"{np.mean([cer(f['heard'], ' '.join(f['sents'])) for f in fs]):.3f} |")

    L += ["", "## Very short first sentence", "",
          "separate = two requests as served, merged = one request. Heard = Whisper has the short word; total = seconds of audio;",
          "gap = silence between the short word and the next sentence; pitch/level step across it.", "",
          "| case | variant | heard | total s | lead s | gap s | pitch step | level step |", "|---|---|---|---|---|---|---|---|"]
    for case in sorted({f["case"] for f in files if f["set"] == "short"}):
        for v in ("separate", "merged", "alone", "separate+trim", "merged+trim"):
            fs = [f for f in files if f["set"] == "short" and f["case"] == case and f["variant"] == v and f["m"]]
            if not fs:
                continue
            word = re.sub(r"\W", "", fs[0]["sents"][0]).lower()
            ok = sum(re.sub(r"[^\w\s]", "", f["heard"].lower()).split()[:1] == [word] for f in fs)
            joins = [j for f in fs for j in f["m"]["joins"]]
            L.append(f"| {fs[0]['sents'][0]} | {v} | {ok}/{len(fs)} | {med([f['m']['dur_s'] for f in fs]):.2f} | {med([f['m']['lead_s'] for f in fs]):.2f} | "
                     + (f"{med([j['gap_s'] for j in joins]):.2f} | {med([j['f0_step'] for j in joins]):+.1f} | {med([abs(j['level_step']) for j in joins]):.1f} |" if joins else "| | |"))

    L += ["", "## colok-words: plain vs \"ch\" spelling", "",
          "c = the phoneme recogniser hears \"col\" (tʃ), not \"kol\"/\"tol\"; Whisper = its spelling of the word.", "",
          "| word | spelling | c heard | heard as | Whisper wrote |", "|---|---|---|---|---|"]
    from collections import Counter
    for w in dict.fromkeys(f["word"] for f in files if f["set"] == "lexicon"):
        for v in ("plain", "ch"):
            fs = [f for f in files if f["set"] == "lexicon" and f["word"] == w and f["variant"] == v]
            heard_as = Counter(f["phones"] for f in fs).most_common(3)
            wrote = Counter(next((t for t in re.sub(r"[^\w\s]", "", f["heard"].lower()).split() if "olok" in t), "?") for f in fs).most_common(3)
            L.append(f"| {w} | {v} | {sum('col' in f['phones'] for f in fs)}/{len(fs)} | "
                     + ", ".join(f"{p} x{n}" for p, n in heard_as) + " | " + ", ".join(f"{p} x{n}" for p, n in wrote) + " |")
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    assert lev("kitten", "sitting") == 3 and syllables("Baik.") == 1 and syllables("Silakan ikuti saya.") == 8
    assert silences(np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0], bool)) == [(3, 5), (6, 7)]
    (measure if sys.argv[1] == "measure" else report)(*sys.argv[2:])

