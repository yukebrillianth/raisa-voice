"""LoRA v3 data (one-off): check the new clips, drop the bad ones, write the manifests, holdouts and test sets.

    # 1. Whisper large-v3-turbo on the 310 new general clips (asr image, server)
    docker run --rm --gpus all -v ~/.cache/huggingface:/hf:ro -e HF_HOME=/hf -v ~/yuke:/yuke \
      -w /yuke/raisa-voice-train/v3 --entrypoint python raisa-voice-asr v3_prep.py asr
    # 2. Gemini (gemini-4-hydrogen) on the new general clips that contain an acronym (laptop, synthetic clips only)
    GEMINI_API_KEY=... python3 train/v3/v3_prep.py gemini ../dataset-baru
    # 3. Manifests (orchestrator image: speech_text)
    docker run --rm -v ~/yuke:/yuke -v ~/yuke/raisa-voice/orchestrator:/orchestrator:ro -w /yuke/raisa-voice-train/v3 \
      -e PYTHONPATH=/orchestrator raisa-voice-orchestrator python v3_prep.py build

Drop rules (decided before looking at the results, applied as written):
- old 600: a clip goes when the audit majority (acronym_audit_600.json, Gemini 3 asks) of any acronym in it is not
  an accepted reading of the user's rules (the `expect` spellings of the acronym manifest; QA in an Indonesian
  sentence must be Indonesian letters). The other 600 rows keep their v2 hybrid transcripts.
- new general 280 + 30: CER > 0.15 against the script, or Whisper hears a run of 2+ words added or missing, or the
  Gemini majority for an acronym in it is not an accepted reading.
- acronym set: only the clips with `ok: true` in .verify.json.
Holdout: v2's 30 clips (never trained; the same 30 are scored) + 1 verified clip per acronym with >= 6 verified clips.
"""
import json, os, random, re, sys
from collections import Counter

D600, DV3 = "/yuke/raisa-voice-data600/", "/yuke/raisa-voice-data-v3/"
OUT = "/yuke/raisa-voice-train/data_v3/"
GENERAL = ["voxcpm2_lora_v3_280_dataset.jsonl", "voxcpm2_lora_v3_tambahan.jsonl"]
ACR = "voxcpm2_lora_v3_acronym.jsonl"
CER_MAX = 0.15


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def chars(t):  # spaces, case and punctuation ignored: "setting-nya" = "settingnya"
    return re.sub(r"[^\w]", "", t.lower())


def words(t):
    return re.sub(r"[^\w\s]", " ", t.lower().replace("-", "")).split()


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def runs(ref, hyp):
    """Word runs Whisper heard that the script lacks (+) or that the script has and Whisper did not hear (-)."""
    import difflib
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
        # a replaced span is a misheard word unless one side is 2+ words longer than the other
        if op == "insert" or (op == "replace" and (j2 - j1) - (i2 - i1) >= 2):
            out.append("+" + " ".join(hyp[j1:j2]))
        if op == "delete" or (op == "replace" and (i2 - i1) - (j2 - j1) >= 2):
            out.append("-" + " ".join(ref[i1:i2]))
    return out


assert lev("kitten", "sitting") == 3
assert runs("a b c d".split(), "a b c d".split()) == [] and runs("a b c d".split(), "a d".split()) == ["-b c"]


def accepted():
    """(acronym, lang) -> accepted readings without spaces, from the acronym manifest."""
    acc = {}
    for r in load(DV3 + ACR):
        for t, a in r["expect"].items():
            acc.setdefault((t, r["lang"]), set()).update(a)
    return acc


def phase_asr():
    import wave
    import numpy as np
    from faster_whisper import WhisperModel

    def load16k(path):
        with wave.open(path) as w:
            rate, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
        return np.interp(np.arange(0, len(x) / rate, 1 / 16000), np.arange(len(x)) / rate, x).astype(np.float32)

    wm = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    out = {}
    ver = json.load(open(DV3 + ACR + ".verify.json"))
    for name in GENERAL + [ACR]:  # the acronym set too: some takes read the generator's pronunciation note aloud
        for r in load(DV3 + name):
            if name == ACR and not ver.get(r["audio"], {}).get("ok"):
                continue
            x = load16k(DV3 + r["audio"])
            _, _, probs = wm.detect_language(x)
            probs = dict(probs)
            # Both languages: a 1-s clip is easily detected wrong ("Siap!" -> English "See ya!"); build keeps the
            # transcript of the language the audio is more likely in, of id and en.
            out[r["audio"]] = {"p": {k: probs.get(k, 0.0) for k in ("id", "en")}}
            for lang in ("id", "en"):
                segs, _ = wm.transcribe(x, language=lang, beam_size=5, condition_on_previous_text=False)
                out[r["audio"]][lang] = " ".join(s.text.strip() for s in segs)
    json.dump(out, open(DV3 + "asr_v3.json", "w"), ensure_ascii=False, indent=0)
    print("transcribed", len(out))


def phase_gemini(root):
    """The acronym check of the verified set, on the new general clips: gemini-4-hydrogen, 3 asks, majority."""
    sys.path.insert(0, os.path.dirname(__file__))
    from concurrent.futures import ThreadPoolExecutor
    from gemini_judge import judge
    acc = {}
    for r in load(root + "/" + ACR):
        for t, a in r["expect"].items():
            acc.setdefault(t, set()).update(a)
    jobs = []
    for name in GENERAL:
        for r in load(root + "/" + name):
            for t in dict.fromkeys(re.findall(r"\b[A-Z]{2,}\b", r["text"])):
                if t in acc:
                    jobs.append((r["audio"], t))
    print(len(jobs), "acronyms to judge", flush=True)
    with ThreadPoolExecutor(8) as pool:
        res = list(pool.map(lambda j: judge(root + "/" + j[0], j[1]), jobs))
    out = {}
    for (audio, t), g in zip(jobs, res):
        out.setdefault(audio, {})[t] = g
    json.dump(out, open(root + "/gemini_v3_general.json", "w"), ensure_ascii=False, indent=0)


def phase_build():
    sys.path.insert(0, "/orchestrator")
    import wave
    import speech_text as st
    from v3_text import one, v3, served_v2
    os.makedirs(OUT, exist_ok=True)
    acc = accepted()
    compact = lambda s: re.sub(r"[^a-z]", "", (s or "").lower())
    dur = lambda p: (lambda w: w.getnframes() / w.getframerate())(wave.open(p))
    report = []

    # --- old 600: v2 split and hybrid transcripts
    rows600 = load(D600 + "voxcpm2_lora_600_dataset.jsonl")
    split = json.load(open(D600 + "split_v2.json"))
    hold600 = split["holdout"]
    hybrid = {r["audio"]: r["text"] for f in ("train_hybrid.jsonl", "holdout_hybrid.jsonl") for r in load(D600 + f)}
    drop600 = {}
    for x in json.load(open(DV3 + "acronym_audit_600.json")):
        heard, n = Counter(x["heard"]).most_common(1)[0]
        lang = st.language(rows600[x["row"]]["text"])
        ok = acc.get((x["token"], lang)) or acc.get((x["token"], "id" if lang == "en" else "en"))
        if x["token"] == "QA" and lang == "id":
            ok = {"kia"}  # not in the manifest; Indonesian letter names by the rule
        if ok and compact(heard) not in ok:
            drop600.setdefault(x["row"], []).append(f"{x['token']} heard '{heard}'")
    train600 = [i for i in range(600) if i not in hold600 and i not in drop600]
    report += ["## Old 600 clips", "", f"Dropped {len(drop600)} (of them {sum(i in hold600 for i in drop600)} in the v2 holdout, "
               "which is never trained and stays scored):", "", "| row | audio | why | text |", "|---|---|---|---|"]
    report += [f"| {i} | {rows600[i]['audio']} | {'; '.join(w)} | {rows600[i]['text']} |" for i, w in sorted(drop600.items())]
    same = sum(v3(rows600[i]["text"]) == hybrid[rows600[i]["audio"]] for i in train600)
    report += ["", f"Kept for training: {len(train600)} with their v2 hybrid transcripts "
               f"({same} are identical to v3_text.v3(); the rest differ only where v2's units/letters were set by hand)."]

    # --- new general clips
    asr = json.load(open(DV3 + "asr_v3.json"))
    gem = json.load(open(DV3 + "gemini_v3_general.json"))
    general, drop_gen, checked = [], {}, []
    for name in GENERAL:
        for r in load(DV3 + name):
            a = asr[r["audio"]]
            # Whisper writes digits and "Prof. Dr.": its text goes through the same normalizer as the script.
            # The language whose transcript is closer to the script: detection alone picks English for 1-s clips
            # ("Baik!" -> "bike", "Siap!" -> "See ya!"); ties go to the detected language.
            ref = chars(r["text"])
            cer_of = {k: lev(ref, chars(v3(a[k]))) / max(1, len(ref)) for k in ("id", "en")}
            lang = min(("id", "en"), key=lambda k: (cer_of[k], -a["p"][k]))
            heard, cer = v3(a[lang]), cer_of[lang]
            extra = [x for x in runs(words(r["text"]), words(heard)) if len(x[1:].split()) >= 2]
            why = []
            if cer > CER_MAX:
                why.append(f"CER {cer:.2f}")
            if extra:
                why.append("words " + ", ".join(extra))
            for t, g in gem.get(r["audio"], {}).items():
                ok = acc.get((t, lang)) or acc.get((t, "id"))  # language of the audio: titles hold "of" in Indonesian rows
                if compact(g["majority"]) not in ok:
                    why.append(f"{t} heard '{g['majority'] or 'no majority'}'")
            checked.append((r["audio"], cer, why, r["text"], heard))
            if why:
                drop_gen[r["audio"]] = why
            else:
                general.append(r)
    cers = sorted(c for _, c, *_ in checked)
    report += ["", "## New general clips (280 + 30 English)", "",
               f"Whisper large-v3-turbo, beam 5, transcribed as id and as en, the one closer to the script kept (1-s clips are detected as English), put through v3_text.v3() (digits, titles). CER over letters and digits (spaces and punctuation "
               f"ignored) against the script: median {cers[len(cers) // 2]:.3f}, 90th percentile {cers[int(len(cers) * 0.9)]:.3f}. "
               f"Gemini acronym check on {len(gem)} clips that contain one.", "",
               f"Dropped {len(drop_gen)}:", "", "| audio | why | script | Whisper heard |", "|---|---|---|---|"]
    report += [f"| {a} | {'; '.join(w)} | {t} | {h} |" for a, c, w, t, h in checked if w]
    report += ["", "Kept with CER > 0.05 (Whisper slips on names and loanwords, listed for the record):", "",
               "| audio | CER | script | Whisper heard |", "|---|---|---|---|"]
    report += [f"| {a} | {c:.2f} | {t} | {h} |" for a, c, w, t, h in checked if not w and c > 0.05]

    # --- acronym set: verified clips only; judge holdout = 1 Indonesian clip per acronym with >= 6 verified clips
    ver = json.load(open(DV3 + ACR + ".verify.json"))
    acr = [r for r in load(DV3 + ACR) if ver.get(r["audio"], {}).get("ok")]
    per = Counter(t for r in acr for t in r["expect"])
    rng = random.Random(3)
    hold_acr = []
    for t in sorted(a for a, n in per.items() if n >= 6):
        pool = [r for r in acr if list(r["expect"]) == [t] and r["lang"] == "id"]
        hold_acr.append(rng.choice(pool))
    hold_ids = {r["audio"] for r in hold_acr}
    acr_train = [r for r in acr if r["audio"] not in hold_ids]
    report += ["", "## Acronym set", "", f"{len(acr)} verified of {len(load(DV3 + ACR))}; judge holdout {len(hold_acr)} "
               f"(one Indonesian clip each of {', '.join(t for r in hold_acr for t in r['expect'])}); trained {len(acr_train)}."]

    # --- manifests: absolute paths inside the training container (/d600, /dv3)
    def row(audio_root, r, text):
        out = {"audio": audio_root + r["audio"], "text": text}
        if r.get("ref_audio"):
            out["ref_audio"] = audio_root + r["ref_audio"]
        return out

    train = [row("/d600/", rows600[i], hybrid[rows600[i]["audio"]]) for i in train600]
    train += [row("/dv3/", r, v3(r["text"])) for r in general]
    train += [row("/dv3/", r, v3(r["text"])) for r in acr_train]
    holdout = [row("/d600/", rows600[i], hybrid[rows600[i]["audio"]]) for i in hold600]
    holdout += [row("/dv3/", r, v3(r["text"])) for r in hold_acr]
    for name, rows in (("train_v3.jsonl", train), ("holdout_v3.jsonl", holdout)):
        with open(OUT + name, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    changed = [(r["text"], v3(r["text"])) for r in general + acr_train if v3(r["text"]) != r["text"]]
    host = lambda p: p.replace("/d600/", D600).replace("/dv3/", DV3)
    secs = lambda rows: sum(dur(host(r["audio"])) for r in rows)
    parts = {"old 600": train[:len(train600)], "new general": train[len(train600):len(train600) + len(general)],
             "acronym": train[len(train600) + len(general):]}
    report += ["", "## Training set", "", "| part | clips | minutes | with ref_audio | longest s |", "|---|---|---|---|---|"]
    for k, rows in list(parts.items()) + [("total", train), ("holdout", holdout)]:
        longest = max(dur(host(r["audio"])) for r in rows)
        report.append(f"| {k} | {len(rows)} | {secs(rows) / 60:.1f} | {sum('ref_audio' in r for r in rows)} "
                      f"({sum('ref_audio' in r for r in rows) / len(rows):.0%}) | {longest:.1f} |")
    report += ["", f"Transcripts changed by v3_text.v3() (new clips): {len(changed)}", ""]
    report += [f"- `{a}` -> `{b}`" for a, b in changed[:40]]
    open(OUT + "prep_report.md", "w").write("# LoRA v3 data\n\n" + "\n".join(report) + "\n")

    # --- evaluation split: v2's 30 holdout + 15 unseen, plus long sentences for the level drop; product text per model
    from served_ab import LENGTH
    sents = []
    for s in split["sentences"]:
        sents.append({**s, "v2": served_v2(s["orig"]), "v3": v3(s["orig"])})
    for secs_, pair in LENGTH.items():
        for lang, t in zip(("id", "en"), pair):
            sents.append({"sid": f"l{secs_}{lang}", "orig": t, "lang": lang, "v2": served_v2(t), "v3": v3(t)})
    json.dump({"sentences": sents, "holdout": hold600}, open(OUT + "split_v3eval.json", "w"), ensure_ascii=False, indent=1)

    # --- acronym / word test set (v3_acro.py) with both product texts; judge-holdout rows as a third carrier kind
    from v3_acro import cases
    tests = cases(hold_acr)
    train_text = " ".join(r["text"] for r in train).lower()
    for c in tests:
        assert c["kind"] == "hold" or c["orig"].lower() not in train_text, c["orig"]
        c["v2"], c["v3"] = served_v2(c["orig"]), c.get("v3") or v3(c["orig"])
    json.dump(tests, open(OUT + "acro_tests.json", "w"), ensure_ascii=False, indent=1)
    # --- A/B listening pack: one request per sentence, played back to back as the robot does
    ab = {"acro_id": ["Silakan bayar pakai QRIS, lalu kirim bukti PDF lewat email sebelum pukul 16.00 WIB.",
                      "Server ini memakai GPU dengan memori 48 GB dan kabel HDMI di meja."],
          "acro_en": ["Please plug the HDMI cable into the USB hub, then open the API guide.", "Welcome to ITS, scan the QR code to sign in."],
          "seekor": ["Ada seekor kucing di lantai keenam, jangan diusir seenaknya ya.", "Colokkan kabelnya ke stopkontak di dinding."],
          "long_reply": ["Teaching Factory adalah tempat mahasiswa belajar membuat robot.",
                         "Di lantai satu ada lab robotika dan ruang produksi, sedangkan di lantai dua ada ruang rapat dan area uji coba "
                         "yang biasanya dipakai tim robot sepak bola untuk berlatih setiap sore.",
                         "Kalau Anda datang bersama rombongan, silakan daftar dulu di meja depan supaya petugas bisa menyiapkan pemandu.",
                         "Ada yang ingin Anda lihat lebih dulu?"],
          "short": ["Baik, saya antar sekarang."],
          "title": ["Pembicara hari ini adalah Prof. Dr. Ir. Budi Santoso, S.T., M.Sc. dari ITS."]}
    json.dump({k: {"orig": v, "v2": [st.normalize(x) for x in v], "v3": [one(x) for x in v]} for k, v in ab.items()},
              open(OUT + "ab_tests.json", "w"), ensure_ascii=False, indent=1)
    print(open(OUT + "prep_report.md").read())
    print(len(tests), "acronym/word test sentences; examples:")
    for c in tests[::25]:
        print(" ", c["kind"], c["token"], "|", c["v2"], "|", c["v3"])


if __name__ == "__main__":
    {"asr": phase_asr, "build": phase_build}.get(sys.argv[1], lambda: phase_gemini(sys.argv[2]))()
