"""Agreement of cheaper / batched judge setups with the reference (gemini-4-hydrogen, 2 single-clip asks)
on clips from out/acro_spell. Verdict = heard reading equals the right reading (letters only)."""
import base64, collections, json, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
O = "/Users/yukebrillianth/IRIS/iohits-tts-v1/raisa-voice/out/acro_spell/"
RIGHT = {"WIB": {"weibe"}, "HDMI": {"hadeemi"}, "API": {"eipiai"}}
norm = lambda s: "".join(c for c in str(s).lower() if c.isalpha())
h = json.load(open(O + "heard.json")); g = json.load(open(O + "gemini.json"))
pool = collections.defaultdict(list)
for x in h:
    ref = norm(g[x["file"]]["majority"]) in RIGHT[x["acronym"]]
    pool[(x["acronym"], ref)].append(x)
random.seed(7)
sample = [x for k in sorted(pool) for x in random.sample(pool[k], min(20, len(pool[k])))]  # balanced right/wrong
random.shuffle(sample)
model, think, batch = sys.argv[1], sys.argv[2], int(sys.argv[3])

def call(group):
    parts = []
    for i, x in enumerate(group):
        parts += [{"text": f"Rekaman {i+1}:"}, {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(open(O + x["file"], "rb").read()).decode()}}]
    parts.append({"text": f"Ada {len(group)} rekaman bernomor. Untuk tiap rekaman, tulis bunyi singkatan yang diminta persis seperti "
                  "terdengar, suku kata demi suku kata, ejaan Indonesia (contoh: \"ji pi yu\", \"ge pe u\", \"ha de em i\"). Jangan membetulkan.\n"
                  + "\n".join(f"Rekaman {i+1}: {x['acronym']}" for i, x in enumerate(group))
                  + '\nJSON saja: {"1": "bunyi", "2": "bunyi", ...}'})
    cfg = {"temperature": 0, "responseMimeType": "application/json"}
    if think != "none": cfg["thinkingConfig"] = {"thinkingLevel": think}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps({"contents": [{"parts": parts}], "generationConfig": cfg}).encode(),
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"})
    for a in range(4):
        try:
            out = json.load(urllib.request.urlopen(req, timeout=240))
            return json.loads(out["candidates"][0]["content"]["parts"][-1]["text"]), out.get("usageMetadata", {})
        except Exception:
            time.sleep(5 * (a + 1))
    return {}, {}

groups = [sample[i:i + batch] for i in range(0, len(sample), batch)]
with ThreadPoolExecutor(6) as ex:
    res = list(ex.map(call, groups))
agree = n = 0; tok = collections.Counter(); conf = collections.Counter()
for group, (ans, u) in zip(groups, res):
    for k in ("promptTokenCount", "thoughtsTokenCount", "candidatesTokenCount"): tok[k] += u.get(k, 0) or 0
    for i, x in enumerate(group):
        ref = norm(g[x["file"]]["majority"]) in RIGHT[x["acronym"]]
        mine = norm(ans.get(str(i + 1), "")) in RIGHT[x["acronym"]]
        agree += ref == mine; n += 1; conf[(ref, mine)] += 1
print(f"{model:18s} think={think:5s} batch={batch:2d} calls={len(groups):3d} agree {agree}/{n} = {agree/n:.0%}  "
      f"(ref right & judged wrong {conf[(True, False)]}, ref wrong & judged right {conf[(False, True)]})  "
      f"tokens/clip: prompt {tok['promptTokenCount']/n:.0f} thoughts {tok['thoughtsTokenCount']/n:.0f} out {tok['candidatesTokenCount']/n:.0f}")
