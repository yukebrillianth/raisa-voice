"""Is gemini-4-hydrogen with thinking turned down still a usable acronym judge? One ask per file on the files the user
labelled by ear (acro_judge.py "labelled"), graded like the reports; prints the token usage of every call.

    GEMINI_API_KEY=... python3 train/v3/gemini_lowthink_check.py out/      # laptop; about 9 calls
First call: the same file without a thinking config (baseline thinking tokens); then the lowest thinking setting the
model accepts (a rejected config is an HTTP 400, not billed) for all 7 files. Writes <out>/gemini_lowthink_check.json.
"""
import base64, json, os, sys, urllib.error, urllib.request

from acro import g2p, grade
from gemini_judge import MODEL, PROMPT


def gphones(bunyi):  # as v3_ana.acro: Gemini's spelling as phones ("kyu ris" = "kyuris", "eich" = "eic")
    text = bunyi.lower()
    for a, b in (("tch", "c"), ("ch", "c"), ("sh", "sy"), ("ee", "i"), ("oo", "u")):
        text = text.replace(a, b)
    return g2p(text)


KEY, O = os.environ["GEMINI_API_KEY"], sys.argv[1]
FILES = [  # (file, {acronym: (rule reading, the user's label)}), from acro_judge.py "labelled"
    ("lora_v2_final/02_wib_final.wav", {"WIB": ("we i be", "wrong")}),
    ("lora_v2_final/05_gpu_gb_final.wav", {"GPU": ("ji pi yu", "wrong"), "GB": ("ji bi", "wrong")}),
    ("lora_v2_final/06_hdmi_its_final.wav", {"HDMI": ("ha de em i", "wrong"), "ITS": ("i te es", "right")}),
    ("lora_v2_ab/u10_hdmi_its_hybrid.wav", {"HDMI": ("ha de em i", "right"), "ITS": ("i te es", "right")}),
    ("lora_v2_ab/h29_dewi_wib_norm.wav", {"WIB": ("we i be", "right")}),
    ("lora_v2_ab/u8_usb_tv_hybrid.wav", {"USB": ("yu es bi", "right"), "TV": ("ti vi", "right")}),
    ("lora_v2_ab/u0_qris_v1.wav", {"QRIS": ("kyuris", "right")}),
]
CONFIGS = [{"thinkingLevel": "minimal"}, {"thinkingLevel": "low"}, {"thinkingBudget": 0}, {"thinkingBudget": 128}]


def call(path, tokens, thinking):
    gen = {"temperature": 0, "responseMimeType": "application/json", **({"thinkingConfig": thinking} if thinking else {})}
    body = {"contents": [{"parts": [{"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(open(path, "rb").read()).decode()}},
                                    {"text": PROMPT.format(tokens=", ".join(tokens))}]}], "generationConfig": gen}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
                                 data=json.dumps(body).encode(), headers={"x-goog-api-key": KEY, "Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=120))
    return json.loads(out["candidates"][0]["content"]["parts"][-1]["text"]), out.get("usageMetadata", {})


log, calls = [], 0
f0, t0 = FILES[0]
answer, usage = call(O + f0, list(t0), None)
calls += 1
log.append({"file": f0, "thinking": None, "answer": answer, "usage": usage})
print("baseline", usage, flush=True)
chosen = None
for cfg in CONFIGS:
    try:
        answer, usage = call(O + f0, list(t0), cfg)
        calls += 1
        chosen = cfg
        log.append({"file": f0, "thinking": cfg, "answer": answer, "usage": usage})
        print("accepted", cfg, usage, flush=True)
        break
    except urllib.error.HTTPError as e:
        print("rejected", cfg, e.code, e.read()[:200], flush=True)
agree = total = 0
for f, toks in FILES:
    if f != f0:
        answer, usage = call(O + f, list(toks), chosen)
        calls += 1
        log.append({"file": f, "thinking": chosen, "answer": answer, "usage": usage})
    answer = next(x["answer"] for x in log if x["file"] == f and x["thinking"] == chosen)
    for t, (rule, label) in toks.items():
        a = answer.get(t) or {}
        heard = str(a.get("bunyi", "")) if isinstance(a, dict) else ""
        verdict = "right" if heard and grade(gphones(heard), g2p(rule), [], []) != "wrong" else "wrong"
        total += 1
        agree += verdict == label
        print(f"{f:40s} {t:5s} user {label:5s} gemini {verdict:5s} heard '{heard}'", flush=True)
print(f"agrees with the user's ears on {agree}/{total}; calls {calls}")
json.dump({"chosen": chosen, "agree": agree, "total": total, "calls": calls, "log": log}, open(O + "gemini_lowthink_check.json", "w"),
          ensure_ascii=False, indent=1)
