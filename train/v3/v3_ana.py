"""Score the v3 renders (v3_gen.py) against the served v2.

    python v3_ana.py fade <hold dir>                    # level drop, end vs start of each clip (numpy)
    python v3_ana.py hold <hold dir> <report.md>        # + v2_ana.py measure/asr output -> table, candidates.txt
    python v3_ana.py phones <acro dir>                  # phoneme recogniser on every acro clip (tts image, GPU)
    GEMINI_API_KEY=... python3 v3_ana.py gemini <acro dir>   # gemini-4-hydrogen, 3 asks (laptop; synthetic clips only)
    python3 v3_ana.py acro <acro dir> <report.md>       # both judges -> tables (plain python)
"""
import glob, json, os, re, sys, wave
from collections import Counter

M = os.environ.get("V3DATA", "/yuke/raisa-voice-train/data_v3/")
D600 = "/yuke/raisa-voice-data600/"


def fade(path, secs=2.5):
    """Level (dB) of the last `secs` of speech minus the first `secs` (muffle.tail_stats' fade, without the STFT)."""
    import numpy as np
    with wave.open(path) as w:
        rate, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64) / 32768
    hop = rate // 100
    e = (x[: len(x) // hop * hop].reshape(-1, hop) ** 2).mean(1) + 1e-12
    db = 10 * np.log10(e)
    speech = np.flatnonzero(db > db.max() - 35)
    first, last = speech[speech <= speech[0] + secs * 100], speech[speech >= speech[-1] - secs * 100]
    return float(10 * np.log10(e[last].mean() / e[first].mean()))


def hold(E, path):
    import numpy as np
    split = json.load(open(M + "split_v3eval.json"))
    S = {s["sid"]: s for s in split["sentences"]}
    meas, asr, fd = (json.load(open(f"{E}/{n}.json")) for n in ("meas", "asr", "fade"))

    def words(t): return re.sub(r"[^\w\s]", " ", t.lower()).split()

    def lev(a, b):
        d = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            p, d[0] = d[0], i
            for j, y in enumerate(b, 1):
                p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
        return d[-1]
    rows = {}
    for r, text in asr.items():
        model, base = r.split("/")
        sid = base.rsplit("_", 1)[0]
        hyp, ref = words(text), words(S[sid]["orig"])
        real = meas["real"].get(sid)
        rows.setdefault(model, []).append({"sid": sid, "cer": lev(" ".join(hyp), " ".join(ref)) / max(1, len(" ".join(ref))),
                                           "wer": lev(hyp, ref) / max(1, len(ref)), "fade": fd["gen"][r],
                                           "tempo": meas["gen"][r]["dur"] / real["dur"] if real else np.nan, **meas["gen"][r]})
    a = lambda xs, k: np.array([x[k] for x in xs], dtype=float)
    summ = {}
    for model, r in rows.items():
        h, u = [x for x in r if x["sid"][0] == "h"], [x for x in r if x["sid"][0] == "u"]
        lg = [x for x in r if x["sid"][0] == "l" and int(re.sub(r"\D", "", x["sid"])) >= 12]
        on = a(h, "onset")
        s = summ[model] = dict(cer=a(h, "cer").mean(), wer=a(h, "wer").mean(), tempo=np.nanmean(a(h, "tempo")),
                               tempo_sd=np.nanstd(a(h, "tempo")), onset=np.nanmedian(on), onset_hi=int((on > 1.3).sum()), n=len(on),
                               f0=np.nanmedian(a(h, "f0")), fade=np.median(a(h, "fade")), cer_u=a(u, "cer").mean(),
                               wer_u=a(u, "wer").mean(), cer_l=a(lg, "cer").mean(), fade_l=np.median(a(lg, "fade")),
                               dur_l=np.mean(a(lg, "dur")))
        # same weights as v2_ana.py (CER holdout + unseen, tempo off 1.0, share of onsets > 1.3x)
        s["score"] = s["cer"] + s["cer_u"] + 0.5 * abs(s["tempo"] - 1) + 0.5 * s["onset_hi"] / s["n"]
    v3 = sorted((m for m in summ if not m.startswith("v2")), key=lambda m: summ[m]["score"])
    cands = v3[:3]
    late = [m for m in v3 if m >= sorted(v3)[len(v3) // 2]] if v3 else []  # names sort by step (v3_s0068 ...)
    if late and late[0] not in cands:
        cands.append(late[0])  # acronyms may need more epochs than the holdout shows: the best of the second half too
    real = list(meas["real"].values())
    rf = [fd["real"][k] for k in fd["real"]]
    L = ["Real holdout recordings: f0 median %.0f Hz, onset median %.2f, >1.3x %d/%d, fade median %+.1f dB" % (
             np.nanmedian([x["f0"] for x in real]), np.nanmedian([x["onset"] for x in real]),
             sum(x["onset"] > 1.3 for x in real), len(real), np.median(rf)), "",
         "Holdout 30 x 3 samples (CER/WER vs the original text, Whisper large-v3-turbo); unseen = 15 new sentences x 3; long = "
         "the 12/20/30 s sentences of served_ab.LENGTH, id + en, x 3. fade = level of the last 2.5 s of speech minus the first "
         "2.5 s (dB, median), with the served Leveller. score = CER + unseen CER + 0.5*|tempo-1| + 0.5*share(onset>1.3).", "",
         "| model | CER | WER | tempo | onset | onset >1.3x | f0 | fade | unseen CER | unseen WER | long CER | long fade | long s | score |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in sorted(summ, key=lambda c: (not c.startswith("v2"), c)):
        s = summ[m]
        L.append(f"| {m}{' **cand**' if m in cands else ''} | {s['cer']:.3f} | {s['wer']:.3f} | {s['tempo']:.2f}±{s['tempo_sd']:.2f} | "
                 f"{s['onset']:.2f} | {s['onset_hi']}/{s['n']} | {s['f0']:.0f} | {s['fade']:+.1f} | {s['cer_u']:.3f} | {s['wer_u']:.3f} | "
                 f"{s['cer_l']:.3f} | {s['fade_l']:+.1f} | {s['dur_l']:.1f} | {s['score']:.3f} |")
    L += ["", "Candidates for the acronym test: " + " ".join(cands)]
    open(path, "w").write("\n".join(L) + "\n")
    open(f"{E}/candidates.txt", "w").write(" ".join(cands) + "\n")
    json.dump(summ, open(f"{E}/summary.json", "w"), indent=1, default=float)
    print("\n".join(L))


def tests():
    return {c["id"]: c for c in json.load(open(M + os.environ.get("TESTS", "acro_tests.json")))}


def files(E):
    T = tests()
    for f in sorted(glob.glob(f"{E}/*/*.wav")):
        model, base = f[len(E) + 1:].split("/")
        yield f, model, T[base[:-4].rsplit("_", 1)[0]]


def phones(E):
    from phones import Phones
    p = Phones()
    out = {f[len(E) + 1:]: p(f) for f, _, _ in files(E)}
    json.dump(out, open(f"{E}/heard_p.json", "w"), ensure_ascii=False)
    print("phones", len(out))


def gemini(E):
    from concurrent.futures import ThreadPoolExecutor
    from gemini_judge import PROMPT, WORD_PROMPT, judge
    path = f"{E}/gemini.json"
    done = json.load(open(path)) if os.path.exists(path) else {}
    todo = [(f, c) for f, _, c in files(E) if f[len(E) + 1:] not in done]
    print(len(done), "answered,", len(todo), "to ask", flush=True)
    def ask(fc):
        try:
            return judge(fc[0], fc[1]["token"], WORD_PROMPT if fc[1]["kind"] == "word" else PROMPT)
        except Exception as e:  # quota (429) after gemini_judge's retries: left unanswered, a rerun asks again
            print("skip", fc[0], e, flush=True)
            return None
    with ThreadPoolExecutor(int(os.environ.get("JUDGE_THREADS", "8"))) as pool:
        for i, ((f, _), r) in enumerate(zip(todo, pool.map(ask, todo)), 1):
            if r is not None:
                done[f[len(E) + 1:]] = r
            if i % 100 == 0 or i == len(todo):
                json.dump(done, open(path, "w"), ensure_ascii=False, indent=0)
                print(i, flush=True)


def word_ok_g(word, heard):
    h = re.sub(r"[^a-z]", "", (heard or "").lower().replace("é", "e"))
    return bool(h) and (h.startswith("c") if word.startswith("colok") else "ee" in h)


def acro(E, path):
    from acro import g2p, grade, heard, region

    def gphones(bunyi):  # Gemini's spelling as phones, as gemini_judge.phones ("kyu ris" = "kyuris", "eich" = "eic")
        text = bunyi.lower()
        for a, b in (("tch", "c"), ("ch", "c"), ("sh", "sy"), ("ee", "i"), ("oo", "u")):
            text = text.replace(a, b)
        return g2p(text)
    P = json.load(open(f"{E}/heard_p.json"))
    G = json.load(open(f"{E}/gemini.json")) if os.path.exists(f"{E}/gemini.json") else {}
    rows = []
    for f, model, c in files(E):
        rel = f[len(E) + 1:]
        secs = (os.path.getsize(f) - 44) / 96000
        got = region(heard(P[rel]), c["left"], c["targets"][0], c["right"])
        if c["kind"] == "word" and c["token"] != "colokkan":  # se-/ke-: both e's must be heard, no slip allowed
            p_ok = got == c["targets"][0] or "".join(got) == "".join(c["targets"][0])
        else:
            p_ok = secs <= 15 and any(grade(got, t, c["left"], c["right"], c["lang"]) != "wrong" for t in c["targets"])
        g = G.get(rel)
        maj = g["majority"] if g else None
        if g is None:
            g_ok = None
        elif c["kind"] == "word":
            g_ok = word_ok_g(c["token"], maj)
        else:
            g_ok = bool(maj) and any(grade(gphones(maj), t, [], [], c["lang"]) != "wrong" for t in c["targets"])
        variant = "override" if c.get("only") else "plain"
        rows.append({"model": model, "id": c["id"], "kind": c["kind"], "token": c["token"], "lang": c["lang"],
                     "variant": variant, "p": p_ok, "g": g_ok, "p_heard": "".join(got), "g_heard": maj or "tidak pasti"})
    json.dump(rows, open(f"{E}/graded.json", "w"), ensure_ascii=False, indent=0)
    models = sorted({r["model"] for r in rows}, key=lambda m: (not m.startswith("v2"), m))
    judged = [r for r in rows if r["g"] is not None]
    L = []
    if judged:
        po = sum(r["p"] == r["g"] for r in judged) / len(judged)
        pp, pg = sum(r["p"] for r in judged) / len(judged), sum(r["g"] for r in judged) / len(judged)
        pe = pp * pg + (1 - pp) * (1 - pg)
        L += [f"Judges agree on {po:.0%} of {len(judged)} clips (Cohen's kappa {(po - pe) / (1 - pe):.2f}).", ""]

    def frac(rs, key):
        rs = [r for r in rs if r[key] is not None]
        return (sum(r[key] for r in rs), len(rs))

    def pct(rs, key):
        a, b = frac(rs, key)
        return f"{a / b:.0%} ({a}/{b})" if b else "-"

    def low(rs, key):  # acronyms under 7/8
        per = {}
        for r in rs:
            if r[key] is not None:
                per.setdefault(r["token"], []).append(r[key])
        return sum(sum(v) < 0.875 * len(v) for v in per.values()), len(per)

    sets = [("Indonesian, 54 acronyms x 2 carriers x 4", lambda r: r["kind"] in ("acro", "new")),
            ("of them: the 22 of acro_report.md", lambda r: r["kind"] == "acro"),
            ("English carriers (HDMI ITS QR API) x 2 x 4", lambda r: r["kind"] == "en"),
            ("judge holdout (19 acronym clips never trained) x 4", lambda r: r["kind"] == "hold")]
    L += ["## Acronym accuracy, both judges", "", "P = phoneme judge (acro.grade: rule reading or one vowel-colour slip), "
          "G = gemini-4-hydrogen majority of 3 asks, graded the same way. Cell: right share (right/clips); acronyms under 7/8.", "",
          "| set | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for name, sel in sets:
        cells = []
        for m in models:
            rs = [r for r in rows if r["model"] == m and sel(r)]
            lp, lg = low(rs, "p"), low(rs, "g")
            cells.append(f"P {pct(rs, 'p')}, {lp[0]}/{lp[1]} under 7/8<br>G {pct(rs, 'g')}, {lg[0]}/{lg[1]} under 7/8")
        L.append(f"| {name} | " + " | ".join(cells) + " |")

    def cell(rs):
        if not rs:
            return "-"
        bad_p = Counter(r["p_heard"] for r in rs if not r["p"]).most_common(1)
        bad_g = Counter(r["g_heard"] for r in rs if r["g"] is False).most_common(1)
        a, b = frac(rs, "p")
        text = f"P {a}/{b}" + (f" `{bad_p[0][0]}`" if bad_p else "")
        a, b = frac(rs, "g")
        return text + (f"<br>G {a}/{b}" + (f" `{bad_g[0][0]}`" if bad_g else "") if b else "")

    for lang, kinds, title in (("id", ("acro", "new"), "Indonesian carriers"), ("en", ("en",), "English carriers"),
                               ("id", ("hold",), "Judge holdout clips")):
        tokens = sorted({r["token"] for r in rows if r["kind"] in kinds and r["lang"] == lang})
        L += ["", f"## Per acronym: {title}", "", "| acronym | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for t in tokens:
            L.append(f"| {t} | " + " | ".join(cell([r for r in rows if r["model"] == m and r["token"] == t and r["kind"] in kinds
                                                     and r["lang"] == lang]) for m in models) + " |")
    L += ["", "## Special words (8 samples each; acronyms = their 2 Indonesian carriers x 4)", "",
          "Words: P = the word's phones exactly (both e's of se-/ke-), G = Gemini syllables keep both e's / start with c.", "",
          "| word | text sent | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    for w in ("seekor", "keenam", "seenaknya", "colokkan"):
        for variant in ("plain", "override"):
            rs = [r for r in rows if r["kind"] == "word" and r["token"] == w and r["variant"] == variant]
            if rs:
                label = "served v2 / raw v3" if variant == "plain" else "respelled (v3 only)"
                L.append(f"| {w} | {label} | " + " | ".join(cell([r for r in rs if r["model"] == m]) or "-" for m in models) + " |")
    for t in ("MAGE", "IRIS", "SKEM", "WIB", "QRIS"):
        L.append(f"| {t} | served v2 / raw v3 | " + " | ".join(cell([r for r in rows if r["model"] == m and r["token"] == t and r["kind"] in ("acro", "new")]) for m in models) + " |")
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    mode, E = sys.argv[1], sys.argv[2].rstrip("/")
    if mode == "fade":
        split = json.load(open(M + "split_v3eval.json"))
        real = {s["sid"]: fade(D600 + s["audio"]) for s in split["sentences"] if s["sid"][0] == "h"}
        gen = {f[len(E) + 1:]: fade(f) for f in sorted(glob.glob(E + "/*/*.wav"))}
        json.dump({"gen": gen, "real": real}, open(E + "/fade.json", "w"))
    elif mode == "hold":
        hold(E, sys.argv[3])
    elif mode == "phones":
        phones(E)
    elif mode == "gemini":
        gemini(E)
    elif mode == "acro":
        acro(E, sys.argv[3])
