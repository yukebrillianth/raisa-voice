"""Keeps only the generated acronym clips whose pronunciation a judge confirms.

    GEMINI_API_KEY=... python3 train/v3/acronym_verify.py ../dataset-baru voxcpm2_lora_v3_acronym.jsonl

For every clip that exists and is not verified yet: ask gemini-4-hydrogen three times how each
acronym was actually pronounced, take the majority, and compare with the manifest's `expect`.
A clip passes only if every acronym in it passes. Failing clips are DELETED so that rerunning
generator/generate.mjs (it skips existing files) makes a new take. Results accumulate in
<manifest>.verify.json, so the script can be rerun after each generation round.

gemini-3.8-flash must not be used as the judge: it writes what the text says, not what it hears.
"""

import base64
import collections
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT, MANIFEST = sys.argv[1], sys.argv[2]
KEY, MODEL = os.environ["GEMINI_API_KEY"], "gemini-4-hydrogen"
STATE = os.path.join(ROOT, MANIFEST + ".verify.json")
PROMPT = """Ini rekaman satu kalimat. Tugasmu menilai PELAFALAN, bukan isi.
Kalimat tertulisnya: "{text}"
Untuk tiap singkatan berikut: {tokens}
tulis bunyi yang BENAR-BENAR terdengar, suku kata demi suku kata, dengan ejaan bahasa Indonesia
(contoh gaya penulisan: "ji pi yu", "ge pe u", "gepiu", "ha de em i", "eich di em ai").
Jangan membetulkan dan jangan menebak dari tulisannya. Kalau bunyinya aneh atau ada huruf yang hilang, tulis apa adanya.
Jawab JSON saja: {{"SINGKATAN": {{"bunyi": "..."}}}}"""


def ask(row):
    audio = base64.b64encode(open(os.path.join(ROOT, row["audio"]), "rb").read()).decode()
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "audio/wav", "data": audio}},
            {"text": PROMPT.format(text=row["text"], tokens=", ".join(row["expect"]))},
        ]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        data=json.dumps(body).encode(), headers={"x-goog-api-key": KEY, "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            out = json.load(urllib.request.urlopen(request, timeout=120))
            return json.loads(out["candidates"][0]["content"]["parts"][-1]["text"])
        except Exception:
            time.sleep(5 * (attempt + 1))
    return {}


def compact(answer, token):
    heard = answer.get(token)
    return re.sub(r"[^a-z]", "", str(heard.get("bunyi", "")).lower()) if isinstance(heard, dict) else ""


def judge(row):
    answers = [ask(row) for _ in range(3)]
    heard = {t: [compact(a, t) for a in answers] for t in row["expect"]}
    # majority of the three answers must be one of the accepted spellings
    ok = all(sum(h in accepted for h in heard[t]) >= 2 for t, accepted in row["expect"].items())
    return {"ok": ok, "heard": heard}


rows = [json.loads(line) for line in open(os.path.join(ROOT, MANIFEST), encoding="utf-8")]
state = json.load(open(STATE)) if os.path.exists(STATE) else {}
todo = [r for r in rows if os.path.exists(os.path.join(ROOT, r["audio"])) and not state.get(r["audio"], {}).get("ok")]
print(f"{len(rows)} rows, {sum(v.get('ok', False) for v in state.values())} verified, judging {len(todo)}", flush=True)
with ThreadPoolExecutor(8) as pool:
    for row, result in zip(todo, pool.map(judge, todo)):
        result["takes"] = state.get(row["audio"], {}).get("takes", 0) + 1
        state[row["audio"]] = result
        if not result["ok"]:
            os.remove(os.path.join(ROOT, row["audio"]))  # next generator run makes a new take
            print("REJECT", row["audio"], result["heard"], flush=True)
json.dump(state, open(STATE, "w"), ensure_ascii=False, indent=1)

per = collections.defaultdict(lambda: [0, 0])
for r in rows:
    for t in r["expect"]:
        per[t][1] += 1
        per[t][0] += bool(state.get(r["audio"], {}).get("ok"))
print("verified", sum(v.get("ok", False) for v in state.values()), "of", len(rows))
print("not complete:", ", ".join(f"{t} {a}/{b}" for t, (a, b) in per.items() if a < b) or "none")
