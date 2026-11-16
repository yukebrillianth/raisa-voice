"""Score v1 and the v2 checkpoints, pick the best per variant, write the report and the A/B files.

    python v2_ana.py measure <eval dir>   # train image (librosa): duration, f0, onset (voicestats.py)
    python v2_ana.py asr <eval dir>       # asr image (faster-whisper large-v3-turbo)
    python v2_ana.py report <eval dir> <report.md> <ab dir>
CER/WER are against the ORIGINAL written text (digits, acronyms), as ana_eval.py scores them.
"""
import glob, json, os, re, shutil, sys

PHASE, E = sys.argv[1], sys.argv[2]
D = "/yuke/raisa-voice-data600/"
split = json.load(open(os.environ.get("SPLIT", D + "split_v2.json")))
S = {s["sid"]: s for s in split["sentences"]}
files = sorted(glob.glob(E + "/*/*.wav"))
rel = lambda f: os.path.relpath(f, E)

if PHASE == "measure":
    from voicestats import measure
    real = [D + S[s]["audio"] for s in S if s[0] == "h"]
    m = list(map(measure, files + real))  # serial: a multiprocessing Pool hangs in this container (~0.15 s/file)
    json.dump({"gen": dict(zip(map(rel, files), m[:len(files)])),
               "real": {s: x for s, x in zip([s for s in S if s[0] == "h"], m[len(files):])}}, open(E + "/meas.json", "w"))

elif PHASE == "asr":
    import wave
    import numpy as np
    from faster_whisper import WhisperModel

    def load16k(path):  # this image's PyAV cannot open files for faster-whisper; linear resample is enough for ASR
        with wave.open(path) as w:
            rate, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
        return np.interp(np.arange(0, len(x) / rate, 1 / 16000), np.arange(len(x)) / rate, x).astype(np.float32)
    wm = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    out = {}
    for f in files:
        sid = os.path.basename(f).rsplit("_", 1)[0]
        segs, _ = wm.transcribe(load16k(f), language=S[sid]["lang"], beam_size=1, condition_on_previous_text=False)
        out[rel(f)] = " ".join(s.text.strip() for s in segs)
    json.dump(out, open(E + "/asr.json", "w"), ensure_ascii=False, indent=0)

elif PHASE == "report":
    import numpy as np
    REPORT, AB = sys.argv[3], sys.argv[4]
    meas, asr = json.load(open(E + "/meas.json")), json.load(open(E + "/asr.json"))

    def norm(t): return re.sub(r"[^\w\s]", " ", t.lower()).split()
    def lev(a, b):
        d = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            p, d[0] = d[0], i
            for j, y in enumerate(b, 1):
                p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
        return d[-1]
    assert lev("kitten", "sitting") == 3

    rows = {}
    for r, text in asr.items():
        cfg, base = r.split("/")
        sid = base.rsplit("_", 1)[0]
        hyp, ref = norm(text), norm(S[sid]["orig"])
        m = meas["gen"][r]
        real = meas["real"].get(sid)
        rows.setdefault(cfg, []).append({"sid": sid, "cer": lev(" ".join(hyp), " ".join(ref)) / max(1, len(" ".join(ref))),
                                         "wer": lev(hyp, ref) / max(1, len(ref)),
                                         "tempo": m["dur"] / real["dur"] if real else np.nan, **m})
    summ = {}
    for cfg, r in rows.items():
        h = [x for x in r if x["sid"][0] == "h"]
        u = [x for x in r if x["sid"][0] == "u"]
        ab = [x for x in h if S[x["sid"]]["row"] >= 500]
        a = lambda xs, k: np.array([x[k] for x in xs], dtype=float)
        on = a(h, "onset")
        summ[cfg] = dict(cer=a(h, "cer").mean(), wer=a(h, "wer").mean(), tempo=np.nanmean(a(h, "tempo")),
                         tempo_sd=np.nanstd(a(h, "tempo")), onset=np.nanmedian(on), onset_hi=int((on > 1.3).sum()),
                         n=len(on), f0=np.nanmedian(a(h, "f0")), cer_abbr=a(ab, "cer").mean(),
                         cer_u=a(u, "cer").mean(), wer_u=a(u, "wer").mean())
        s = summ[cfg]
        # shortcut: hand-set weights (CER holdout + unseen, tempo off 1.0, share of onsets > 1.3x); the full table is in the report.
        s["score"] = s["cer"] + s["cer_u"] + 0.5 * abs(s["tempo"] - 1) + 0.5 * s["onset_hi"] / s["n"]
    best = {v: min((c for c in summ if c.startswith(v + "_")), key=lambda c: summ[c]["score"])
            for v in ("norm", "hybrid") if any(c.startswith(v + "_") for c in summ)}
    best["v1"] = "v1"
    real = list(meas["real"].values())

    L = ["# LoRA v2 (600 clips) evaluation", "",
         f"Real holdout recordings: f0 median {np.nanmedian([x['f0'] for x in real]):.0f} Hz, onset ratio median "
         f"{np.nanmedian([x['onset'] for x in real]):.2f}, >1.3x {sum(x['onset'] > 1.3 for x in real)}/{len(real)}", "",
         "Holdout 30 (CER/WER vs original text, Whisper large-v3-turbo, 3 samples each) | cer_abbr = holdout rows 500-599 | unseen = 15 new sentences.",
         "score = CER + unseen CER + 0.5*|tempo-1| + 0.5*share(onset>1.3); lowest per variant is picked.", "",
         "| config | CER | WER | tempo | onset | onset >1.3x | f0 | CER abbr10 | unseen CER | unseen WER | score |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cfg in sorted(summ, key=lambda c: (c != "v1", c)):
        s = summ[cfg]
        mark = " **best**" if cfg in best.values() and cfg != "v1" else ""
        L.append(f"| {cfg}{mark} | {s['cer']:.3f} | {s['wer']:.3f} | {s['tempo']:.2f}±{s['tempo_sd']:.2f} | {s['onset']:.2f} | "
                 f"{s['onset_hi']}/{s['n']} | {s['f0']:.0f} | {s['cer_abbr']:.3f} | {s['cer_u']:.3f} | {s['wer_u']:.3f} | {s['score']:.3f} |")
    L += ["", f"Best: {best}", "", "Never in holdout, trained with the user's readings: " + json.dumps(split["train_only"], ensure_ascii=False),
          f"Holdout rows: {split['holdout']}; train clips: {split['n_train']}", ""]
    if os.path.exists(E + "/times.txt"):
        L += ["Training wall time:", "```", open(E + "/times.txt").read().strip(), "```", ""]

    # A/B: sample 0 of every unseen sentence and every holdout sentence whose two transcripts differ.
    if "norm" not in best:  # a sweep of one checkpoint (cfg/timesteps): table only
        open(REPORT, "w").write("\n".join(L) + "\n"); print("\n".join(L)); sys.exit()
    os.makedirs(AB, exist_ok=True)
    pick = [s for s in S.values() if s["sid"][0] == "u" or s["norm"] != s["hybrid"]]
    idx = ["<sentence>_<v1|norm|hybrid>.wav, sample 0 of 3. Configs: " + json.dumps(best), ""]
    for s in pick:
        tag = "_".join([s["sid"]] + [w.lower() for w in re.findall(r"\b[A-Z][A-Za-z]{1,}\b", s["orig"]) if w.isupper()][:3])
        idx.append(f"{tag}  [{s['lang']}]  written: {s['orig']}")
        for v, cfg in (("v1", "v1"), ("norm", best["norm"]), ("hybrid", best["hybrid"])):
            src = f"{cfg}/{s['sid']}_0.wav"
            shutil.copy(f"{E}/{src}", f"{AB}/{tag}_{v}.wav")
            sent = s["hybrid"] if v == "hybrid" else s["norm"]
            idx.append(f"   {v:6s} TTS: {sent}\n          heard: {asr[src]}")
        idx.append("")
    open(AB + "/index.txt", "w").write("\n".join(idx))
    L += [f"A/B files: {AB} ({len(pick)} sentences x 3), see index.txt"]
    open(REPORT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
