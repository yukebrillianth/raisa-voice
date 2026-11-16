"""How was each acronym read? Phoneme recogniser -> the acronym's phones -> compared with the rule reading.

    python acro_judge.py actor                 # the training recordings: how the data itself reads each acronym
    python acro_judge.py labelled              # the files the user labelled by ear
    python acro_judge.py matrix <dir>          # acro_gen.py output: <dir>/manifest.json -> <dir>/heard.json (all phones)
    python acro_judge.py report <dir> <report.md>   # tables from heard.json and, when there, gemini.json (plain python)
The report cuts the acronym out and grades it with the current acro.py, so a rule change needs no GPU.
Second judge: gemini_judge.py (Gemini listens and writes what it heard); the report shows both and how often they agree.
Grades are acro.grade: "exact", "near" (one vowel-colour slip such as "weibi" for "we i be") or wrong
(any consonant, "gibi" for "ji bi", or "ai" for "i"). "runaway": the sample did not stop within 15 s.
Run in the tts image (torch, transformers, librosa) with the repo at /rv and ~/yuke at /yuke.
"""
import json
import os
import re
import sys
from collections import Counter

from acro import EN, LANG, WORD, cases, g2p, grade, heard, region

D = "/yuke/raisa-voice-data600/"
READINGS = {**EN, **LANG, **WORD}


def split_at(text: str, reading: str):
    m = re.search(rf"\b{re.escape(reading)}\b", text)
    return (g2p(text[: m.start()]), g2p(reading), g2p(text[m.end():])) if m else None


def report(out: str, path: str):
    rows = json.load(open(out + "/heard.json"))
    gemini = json.load(open(out + "/gemini.json")) if os.path.exists(out + "/gemini.json") else {}
    spelling = any(r["form"] not in ("raw", "phon") for r in rows)
    context = {(a, lang, n): (left, target, right) for a, n, lang, _, left, target, right in cases(spellings=spelling)}
    for r in rows:
        left, target, right = context[(r["acronym"], r["lang"], r["carrier"])]
        r["target"], r["heard"] = target, " ".join(region(heard(r["tokens"]), left, target, right))
        r["grade"] = "runaway" if r["secs"] > 15 else grade(r["heard"].split(), target, left, right, r["lang"])
        r["ok"] = r["grade"] in ("exact", "near")
        g = gemini.get(r["file"])
        # Gemini writes Indonesian spelling; no majority among its three answers counts as wrong.
        r["g_heard"] = g["majority"] if g else None
        r["g_ok"] = bool(g["majority"]) and grade(g2p(g["majority"]), target, [], [], r["lang"]) != "wrong" if g else None
    cells = {}
    for r in rows:
        cells.setdefault((r["model"], r["form"], r["temp"], r["lang"], r["acronym"]), []).append(r)
    configs = sorted({k[:3] for k in cells}, key=lambda c: (-float(c[2]), c[0], c[1]))
    judged = [r for r in rows if r["g_ok"] is not None]
    L = ["# Acronym reading consistency", "",
         "P = phoneme judge: wav2vec2-xlsr-53 espeak recogniser (`phones.py`), the acronym's phones cut out by aligning to the",
         "carrier sentence and graded by `acro.grade` (right = exact or one vowel-colour slip). G = Gemini judge",
         "(`gemini_judge.py`: writes what it heard, majority of 3 asks, graded by the same rule). Cell = right / samples",
         "(2 carrier sentences x 8 samples), then the most common wrong reading each judge heard, in Indonesian spelling.", ""]
    if judged:
        both = sum(r["ok"] == r["g_ok"] for r in judged)
        po = both / len(judged)
        pe = (sum(r["ok"] for r in judged) * sum(r["g_ok"] for r in judged)
              + sum(not r["ok"] for r in judged) * sum(not r["g_ok"] for r in judged)) / len(judged) ** 2
        L += [f"Judges agree on {both}/{len(judged)} clips ({po:.0%}, Cohen's kappa {(po - pe) / (1 - pe):.2f}); P right {sum(r['ok'] for r in judged)}, "
              f"G right {sum(r['g_ok'] for r in judged)}, G without a majority {sum(not r['g_heard'] for r in judged)}.", ""]

    def cell(rs):
        text = f"P {sum(r['ok'] for r in rs)}/{len(rs)}"
        bad = Counter(r["heard"].replace(" ", "") for r in rs if not r["ok"]).most_common(1)
        text += f" `{bad[0][0]}`x{bad[0][1]}" if bad else ""
        runaway = sum(r["grade"] == "runaway" for r in rs)
        text += f" runaway {runaway}" if runaway else ""
        gs = [r for r in rs if r["g_ok"] is not None]
        if gs:
            bad = Counter((r["g_heard"] or "tidak pasti") for r in gs if not r["g_ok"]).most_common(1)
            text += f"<br>G {sum(r['g_ok'] for r in gs)}/{len(gs)}" + (f" `{bad[0][0]}`x{bad[0][1]}" if bad else "")
        return text

    for lang in ("id", "en"):
        acronyms = list(dict.fromkeys(k[4] for k in cells if k[3] == lang))
        for temp in [] if not acronyms else sorted({c[2] for c in configs}, reverse=True):
            cols = [c for c in configs if c[2] == temp and any((*c, lang, a) in cells for a in acronyms)]
            if spelling:  # one table per acronym: its spellings are the rows
                for a in acronyms:
                    models = list(dict.fromkeys(c[0] for c in cols))
                    L += [f"## {a} ({''.join(context[(a, lang, 0)][1])}), spellings, temperature {temp}", "",
                          "| written | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
                    for form in dict.fromkeys(c[1] for c in cols if (*c, lang, a) in cells):
                        L.append(f"| {form} | " + " | ".join(cell(cells[(m, form, temp, lang, a)]) for m in models) + " |")
                    L.append("")
                continue
            L += [f"## {'Indonesian' if lang == 'id' else 'English'} sentences, temperature {temp}", "",
                  "| acronym (rule reading) | " + " | ".join(f"{m} {f}" for m, f, _ in cols) + " |", "|---|" + "---|" * len(cols)]
            for a in acronyms:
                L.append(f"| {a} ({''.join(context[(a, lang, 0)][1])}) | " + " | ".join(cell(cells[(*c, lang, a)]) for c in cols) + " |")
            def total(c, key):
                rs = [r for a in acronyms for r in cells[(*c, lang, a)] if r[key] is not None]
                low = sum(sum(r[key] for r in cells[(*c, lang, a)] if r[key] is not None) < 0.875 * len(cells[(*c, lang, a)]) for a in acronyms)
                return f"{sum(r[key] for r in rs) / len(rs):.0%}, {low} under 7/8" if rs else "-"
            L.append("| **all**: right, acronyms under 7/8 | " + " | ".join(
                f"P {total(c, 'ok')}" + (f"<br>G {total(c, 'g_ok')}" if judged else "")
                + f"<br>runaway {sum(r['grade'] == 'runaway' for a in acronyms for r in cells[(*c, lang, a)])}" for c in cols) + " |")
            # The best the model can do with one lexicon: per acronym, the form more often right.
            for key, name in (("ok", "P"), ("g_ok", "G")):
                if key == "g_ok" and not judged:
                    continue
                line = []
                for m in dict.fromkeys(c[0] for c in cols):
                    picks = {a: max((f for mm, f, _ in cols if mm == m), key=lambda f: sum(bool(r[key]) for r in cells[(m, f, temp, lang, a)])) for a in acronyms}
                    rs = [r for a, f in picks.items() for r in cells[(m, f, temp, lang, a)] if r[key] is not None]
                    low = [a for a, f in picks.items() if sum(bool(r[key]) for r in cells[(m, f, temp, lang, a)]) < 0.875 * len(cells[(m, f, temp, lang, a)])]
                    line.append(f"**{m}**: {sum(r[key] for r in rs) / max(1, len(rs)):.0%}; raw for {' '.join(a for a, f in picks.items() if f == 'raw') or '-'}; under 7/8: {' '.join(low) or '-'}")
                L.append(f"| best form per acronym ({name}) | " + " <br> ".join(line) + " |" + " |" * (len(cols) - 1))
            L.append("")
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    mode = sys.argv[1]
    if mode == "report":
        return report(sys.argv[2], sys.argv[3])
    from phones import Phones
    phones = Phones()

    def read(path, left, target, right):
        return " ".join(region(heard(phones(path)), left, target, right))

    if mode == "actor":
        orig = [json.loads(line) for line in open(D + "voxcpm2_lora_600_dataset.jsonl")]
        norm = [json.loads(line) for line in open(D + "voxcpm2_lora_600_dataset.normalized.jsonl")]
        for acronym, reading in READINGS.items():
            got = []
            for o, n in zip(orig, norm):
                parts = split_at(n["text"], reading) if re.search(rf"\b{acronym}\b", o["text"]) else None
                if parts:
                    got.append(read(D + o["audio"], *parts))
            grades = Counter(grade(g.split(), g2p(reading), [], []) for g in got)
            print(f"{acronym:6s} rule [{' '.join(g2p(reading))}] exact {grades['exact']} near {grades['near']} of {len(got)}: " + " | ".join(got), flush=True)

    elif mode == "labelled":
        O = "/rv/out/"
        files = [  # (file, sentence as sent with the rule reading written out, {reading: the user's label})
            ("lora_v2_final/02_wib_final.wav", "Acaranya mulai pukul sembilan we i be di aula utama.", {"we i be": "wrong"}),
            ("lora_v2_final/05_gpu_gb_final.wav", "Server ini memakai ji pi yu dengan memori empat puluh delapan ji bi.", {"ji pi yu": "wrong", "ji bi": "wrong"}),
            ("lora_v2_final/06_hdmi_its_final.wav", "Sambungkan laptop ke proyektor pakai kabel ha de em i di gedung i te es.", {"ha de em i": "wrong", "i te es": "right"}),
            ("lora_v2_ab/u9_gpu_gb_hybrid.wav", "Server ini memakai ji pi yu dengan memori empat puluh delapan ji bi.", {"ji pi yu": "right?", "ji bi": "right?"}),
            ("lora_v2_ab/u10_hdmi_its_hybrid.wav", "Sambungkan laptop ke proyektor pakai kabel ha de em i di gedung i te es.", {"ha de em i": "right", "i te es": "right"}),
            ("lora_v2_ab/h29_dewi_wib_norm.wav", "Ibu Dewi selaku pembina acara akan memberikan sambutan pada pukul sembilan tiga puluh we i be.", {"we i be": "right"}),
            ("lora_v2_ab/u7_pdf_wib_norm.wav", "Kirim berkas pidief lewat email sebelum pukul enam belas empat puluh lima we i be.", {"pidief": "right", "we i be": "right"}),
            ("lora_v2_ab/u8_usb_tv_hybrid.wav", "Cholokkan flashdisk ke port yu es bi di belakang ti vi.", {"yu es bi": "right", "ti vi": "right"}),
            ("lora_v2_ab/u0_qris_v1.wav", "Total belanja Anda satu juta dua ratus tujuh puluh lima ribu rupiah, bisa dibayar pakai kyuris di kasir.", {"kyuris": "right"}),
            ("lora_v2_final/02_wib_v1.wav", "Acaranya mulai pukul sembilan we i be di aula utama.", {"we i be": "-"}),
        ]
        agree = total = 0
        for f, text, readings in files:
            for reading, label in readings.items():
                left, target, right = split_at(text, reading)
                got = region(heard(phones(O + f)), left, target, right)
                verdict = "wrong" if grade(got, target, left, right) == "wrong" else "right"
                total += label in ("right", "wrong")
                agree += label == verdict
                print(f"{f:38s} [{reading}] user: {label:7s} judge: {verdict:5s} heard [{''.join(got)}]", flush=True)
        print(f"judge agrees with the user's label on {agree}/{total}")

    elif mode == "matrix":
        out = sys.argv[2]
        manifest = json.load(open(out + "/manifest.json"))
        for m in manifest:
            m["tokens"] = phones(f"{out}/{m['file']}")  # raw recogniser output; acro.heard is applied by the report
            m["secs"] = (os.path.getsize(f"{out}/{m['file']}") - 44) / 96000  # 48 kHz PCM16
            for key in ("left", "target", "right"):  # the report takes these from acro.py
                m.pop(key, None)
        json.dump(manifest, open(out + "/heard.json", "w"), ensure_ascii=False, indent=0)
        print("judged", len(manifest))


main()
