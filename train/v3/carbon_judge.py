"""Cheap second acronym judge: gemini-4-carbon, thinking "low", 10 clips per request, one answer per clip.

    GEMINI_API_KEY=... python3 carbon_judge.py check <out dir>                 # the user-labelled files first (2 calls)
    GEMINI_API_KEY=... V3DATA=... python3 carbon_judge.py run <acro dir> <model> ...   # Indonesian acronym clips
Measured by the coordinator on 120 reference clips: 86% agreement with the hydrogen 2-ask reference, 0 thinking tokens,
about 170 prompt + 10 output tokens per clip. Every Gemini judge here is noisy (15-20% on borderline readings): the
phoneme judge stays primary. `run` writes <acro dir>/carbon.json {file: heard} and prints the token usage.
"""
import base64, json, os, sys, time, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from acro import g2p, grade

MODEL, BATCH = "gemini-4-carbon", 10


def gphones(bunyi):  # as v3_ana.acro
    text = bunyi.lower()
    for a, b in (("tch", "c"), ("ch", "c"), ("sh", "sy"), ("ee", "i"), ("oo", "u")):
        text = text.replace(a, b)
    return g2p(text)


def call(group):
    """group: [(path, acronym)] -> ({"1": heard, ...}, usage)."""
    parts = []
    for i, (path, _) in enumerate(group):
        parts += [{"text": f"Rekaman {i + 1}:"},
                  {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(open(path, "rb").read()).decode()}}]
    parts.append({"text": f"Ada {len(group)} rekaman bernomor. Untuk tiap rekaman, tulis bunyi singkatan yang diminta persis seperti "
                  "terdengar, suku kata demi suku kata, ejaan Indonesia (contoh: \"ji pi yu\", \"ge pe u\", \"ha de em i\"). Jangan membetulkan.\n"
                  + "\n".join(f"Rekaman {i + 1}: {a}" for i, (_, a) in enumerate(group))
                  + '\nJSON saja: {"1": "bunyi", "2": "bunyi", ...}'})
    cfg = {"temperature": 0, "responseMimeType": "application/json", "thinkingConfig": {"thinkingLevel": "low"}}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
                                 data=json.dumps({"contents": [{"parts": parts}], "generationConfig": cfg}).encode(),
                                 headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            out = json.load(urllib.request.urlopen(req, timeout=240))
            return json.loads(out["candidates"][0]["content"]["parts"][-1]["text"]), out.get("usageMetadata", {})
        except Exception as e:
            print("retry", attempt, e, flush=True)
            time.sleep(10 * (attempt + 1))
    return {}, {}


def batched(items):
    groups = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(call, groups))
    heard, tok, failed = {}, Counter(), 0
    for group, (ans, usage) in zip(groups, res):
        failed += not ans
        for k in ("promptTokenCount", "thoughtsTokenCount", "candidatesTokenCount"):
            tok[k] += usage.get(k, 0) or 0
        for i, item in enumerate(group):
            heard[item] = str(ans.get(str(i + 1), "")) if ans else None
    n = len(items)
    print(f"{len(groups)} calls ({failed} failed), per clip: prompt {tok['promptTokenCount'] / n:.0f}, thinking "
          f"{tok['thoughtsTokenCount'] / n:.0f}, output {tok['candidatesTokenCount'] / n:.0f} tokens", flush=True)
    return heard, len(groups), tok


def check(O):
    labelled = [("lora_v2_final/02_wib_final.wav", "WIB", "we i be", "wrong"),
                ("lora_v2_final/05_gpu_gb_final.wav", "GPU", "ji pi yu", "wrong"), ("lora_v2_final/05_gpu_gb_final.wav", "GB", "ji bi", "wrong"),
                ("lora_v2_final/06_hdmi_its_final.wav", "HDMI", "ha de em i", "wrong"), ("lora_v2_final/06_hdmi_its_final.wav", "ITS", "i te es", "right"),
                ("lora_v2_ab/u10_hdmi_its_hybrid.wav", "HDMI", "ha de em i", "right"), ("lora_v2_ab/u10_hdmi_its_hybrid.wav", "ITS", "i te es", "right"),
                ("lora_v2_ab/h29_dewi_wib_norm.wav", "WIB", "we i be", "right"),
                ("lora_v2_ab/u8_usb_tv_hybrid.wav", "USB", "yu es bi", "right"), ("lora_v2_ab/u8_usb_tv_hybrid.wav", "TV", "ti vi", "right"),
                ("lora_v2_ab/u0_qris_v1.wav", "QRIS", "kyuris", "right")]
    heard, calls, tok = batched([(O + f, a) for f, a, _, _ in labelled])
    agree = 0
    for f, a, rule, label in labelled:
        h = heard[(O + f, a)] or ""
        verdict = "right" if h and grade(gphones(h), g2p(rule), [], []) != "wrong" else "wrong"
        agree += verdict == label
        print(f"{f:40s} {a:5s} user {label:5s} carbon {verdict:5s} heard '{h}'")
    print(f"carbon agrees with the user's ears on {agree}/{len(labelled)}")
    json.dump({"agree": agree, "total": len(labelled), "calls": calls, "tokens": tok}, open(O + "carbon_check.json", "w"), indent=1)


def run(E, models):
    from v3_ana import files
    kinds = os.environ.get("KINDS", "acro,new").split(",")
    items = [(f, c["token"]) for f, m, c in files(E) if m in models and c["kind"] in kinds]
    heard, calls, tok = batched(items)
    path = f"{E}/carbon.json"
    done = json.load(open(path)) if os.path.exists(path) else {}
    done.update({f[len(E) + 1:]: h for (f, _), h in heard.items() if h is not None})
    json.dump(done, open(path, "w"), ensure_ascii=False, indent=0)
    json.dump({"calls": calls, "tokens": tok, "clips": len(items)}, open(f"{E}/carbon_usage_{'_'.join(models)}.json", "w"), indent=1)


def report(dirs):
    """Carbon vs the phoneme judge (graded.json of v3_ana.py acro) per model, Indonesian acronym clips."""
    from v3_ana import tests
    T = tests()

    def under(d):  # acronyms under 7/8
        acc = {}
        for tid, vs in d.items():
            a = acc.setdefault(T[tid]["token"], [0, 0])
            a[0] += sum(vs)
            a[1] += len(vs)
        return sum(r < 0.875 * n for r, n in acc.values()), len(acc)

    L = ["| model | P right | Carbon right | acronyms under 7/8 (P / Carbon) | P and Carbon agree |", "|---|---|---|---|---|"]
    for E in dirs:
        C = json.load(open(f"{E}/carbon.json"))
        rows = [r for r in json.load(open(f"{E}/graded.json")) if r["kind"] in ("acro", "new")]
        for m in sorted({k.split("/")[0] for k in C}):
            P, G = {}, {}
            for r in rows:
                if r["model"] == m:
                    P.setdefault(r["id"], []).append(r["p"])  # file order = sample order, as below
            for k in sorted(k for k in C if k.split("/")[0] == m):
                tid = k.split("/")[1][:-4].rsplit("_", 1)[0]
                h = C[k]
                G.setdefault(tid, []).append(bool(h) and any(grade(gphones(h), t, [], [], T[tid]["lang"]) != "wrong"
                                                             for t in T[tid]["targets"]))
            pairs = [(p, g) for tid in G for p, g in zip(P[tid], G[tid])]
            n, (up, ut), (gp, gt) = len(pairs), under(P), under(G)
            L.append(f"| {m} | {sum(p for p, _ in pairs) / n:.0%} ({sum(p for p, _ in pairs)}/{n}) | {sum(g for _, g in pairs) / n:.0%} "
                     f"({sum(g for _, g in pairs)}/{n}) | {up}/{ut} / {gp}/{gt} | {sum(p == g for p, g in pairs) / n:.0%} |")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report([d.rstrip("/") for d in sys.argv[2:]])
    elif sys.argv[1] == "check":
        check(sys.argv[2])
    else:
        run(sys.argv[2].rstrip("/"), sys.argv[3:])
