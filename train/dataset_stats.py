import json, sys, numpy as np
from multiprocessing import Pool
from voicestats import measure
D = "/home/teaching-factory/yuke/raisa-voice-data/"
rows = [json.loads(l) for l in open(D + "voxcpm2_lora_100_dataset.jsonl")]
hold = set(json.load(open(D + "holdout_idx.json")))
with Pool(24) as p:
    m = p.map(measure, [D + r["audio"] for r in rows])
f0 = np.array([x["f0"] for x in m]); on = np.array([x["onset"] for x in m])
print(f"dataset f0 median {np.nanmedian(f0):.0f} Hz (p10 {np.nanpercentile(f0,10):.0f}, p90 {np.nanpercentile(f0,90):.0f}), onset ratio median {np.nanmedian(on):.2f}, >1.3x {int((on>1.3).sum())}/{len(on)}")
out = {"f0_median": float(np.nanmedian(f0)), "onset_median": float(np.nanmedian(on)),
       "holdout": {rows[i]["audio"]: m[i] for i in hold}}
# Calm reference: a 4-8 s training clip with f0 near the dataset median and the flattest onset.
cands = [(abs(m[i]["f0"] - out["f0_median"]) / out["f0_median"] + abs(m[i]["onset"] - 1.0), i) for i in range(len(rows))
         if i not in hold and 4 <= m[i]["dur"] <= 8 and not np.isnan(m[i]["onset"]) and not any(w in rows[i]["text"].lower() for w in ("please", "your", "the "))]
cands.sort()
for s, i in cands[:3]: print(f"ref candidate {rows[i]['audio']} score {s:.2f} f0 {m[i]['f0']:.0f} onset {m[i]['onset']:.2f} dur {m[i]['dur']:.1f} | {rows[i]['text']}")
out["calm_ref"] = {"audio": rows[cands[0][1]]["audio"], "text": rows[cands[0][1]]["text"]}
for r in ["wavs/ref_01.wav","wavs/ref_02.wav","wavs/ref_03.wav","wavs/ref_04.wav"]:
    x = measure(D + r); print(r, {k: round(v, 2) for k, v in x.items()})
json.dump(out, open(D + "stats.json", "w"), indent=1, ensure_ascii=False)
