"""Score each config: Whisper CER/WER, tempo vs real holdout recordings, onset pitch ratio, f0 median."""
import glob, json, os, re, sys
from multiprocessing import Pool
import numpy as np
from voicestats import measure

E = sys.argv[1]
D = "/home/teaching-factory/yuke/raisa-voice-data/"
stats = json.load(open(D + "stats.json"))
sents = dict(json.load(open(E + "/sentences.json")))
hold = [json.loads(l) for l in open(D + "holdout.jsonl")]
real_dur = {f"h{i}": stats["holdout"][r["audio"]]["dur"] for i, r in enumerate(hold)}

def norm(t): return re.sub(r"[^\w\s]", " ", t.lower()).split()
def lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]

files = sorted(glob.glob(E + "/*/*.wav"))
with Pool(24) as p:
    meas = dict(zip(files, p.map(measure, files)))
from faster_whisper import WhisperModel
wm = WhisperModel("large-v3", device="cuda", compute_type="float16")
EN = {"h8", "h9", "x4"}
rows = {}
for f in files:
    cfg, sid = f.split("/")[-2], os.path.basename(f).split("_")[0]
    segs, _ = wm.transcribe(f, language="en" if sid in EN else "id", beam_size=1, condition_on_previous_text=False)
    hyp, ref = norm(" ".join(s.text for s in segs)), norm(sents[sid])
    m = meas[f]
    rows.setdefault(cfg, []).append({"sid": sid, "cer": lev(" ".join(hyp), " ".join(ref)) / max(1, len(" ".join(ref))),
        "wer": lev(hyp, ref) / max(1, len(ref)), "tempo": m["dur"] / real_dur[sid] if sid in real_dur else np.nan, **m})
print(f"real recordings: f0 median {stats['f0_median']:.0f} Hz, onset ratio median {stats['onset_median']:.2f}")
print(f"{'config':32s} {'CER':>5s} {'WER':>5s} {'tempo':>11s} {'onset':>5s} {'>1.3':>5s} {'f0':>4s} {'f0 sd/sent':>10s}")
summary = {}
for cfg, r in sorted(rows.items()):
    a = lambda k: np.array([x[k] for x in r], dtype=float)
    per_sent = {}
    for x in r: per_sent.setdefault(x["sid"], []).append(x["f0"])
    f0sd = np.nanmean([np.nanstd(v) for v in per_sent.values()])
    t = a("tempo"); on = a("onset")
    summary[cfg] = dict(cer=np.nanmean(a("cer")), wer=np.nanmean(a("wer")), tempo=np.nanmean(t), tempo_sd=np.nanstd(t),
                        onset=np.nanmedian(on), onset_hi=int((on > 1.3).sum()), n=len(on), f0=np.nanmedian(a("f0")), f0sd=f0sd)
    s = summary[cfg]
    print(f"{cfg:32s} {s['cer']:5.3f} {s['wer']:5.3f} {s['tempo']:5.2f}±{s['tempo_sd']:.2f} {s['onset']:5.2f} {s['onset_hi']:2d}/{s['n']:<2d} {s['f0']:4.0f} {s['f0sd']:10.1f}")
json.dump(summary, open(E + "/summary.json", "w"), indent=1)
