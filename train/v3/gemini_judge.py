"""Second acronym judge: Gemini listens to each clip and writes what it heard (train/gemini_judge_probe.py, generalised).

    GEMINI_API_KEY=... python gemini_judge.py <dir> [temp]     # <dir>/heard.json + the wavs -> <dir>/gemini.json
    GEMINI_API_KEY=... python gemini_judge.py --words <dir>    # served_ab.py "seword" set -> <dir>/gemini_words.json
Model gemini-4-hydrogen at temperature 0 (validated on the files the user labelled by ear; gemini-3.8-flash
was not usable, it heard what the text says). Three asks per clip, majority of the heard sound; the third ask
is skipped when the first two agree, which gives the same majority. No majority = "tidak pasti" (None).
Only synthetic test clips are sent. Rerun to resume: answered clips are kept.
"""
import base64, json, os, re, sys, time, urllib.error, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from acro import g2p

MODEL = "gemini-4-hydrogen"
KEY = os.environ["GEMINI_API_KEY"]
PROMPT = """Ini rekaman satu kalimat dari mesin TTS. Tugasmu menilai PELAFALAN, bukan isi.
Untuk tiap singkatan berikut: {tokens}
tulis bunyi yang BENAR-BENAR terdengar, suku kata demi suku kata, dengan ejaan bahasa Indonesia
(contoh gaya penulisan: "ji pi yu", "ge pe u", "gepiu", "ha de em i", "eich di em ai").
Jangan membetulkan dan jangan menebak dari tulisannya. Kalau bunyinya aneh atau ada huruf yang hilang, tulis apa adanya.
Lalu beri label: "inggris" (nama huruf Inggris), "indonesia" (nama huruf Indonesia), "kata" (dibaca sebagai satu kata), atau "kacau".
Jawab JSON saja: {{"SINGKATAN": {{"bunyi": "...", "label": "..."}}}}"""


WORD_PROMPT = """Ini rekaman satu kalimat dari mesin TTS. Tugasmu menilai PELAFALAN, bukan isi.
Untuk kata berikut: {tokens}
tulis bunyi yang BENAR-BENAR terdengar, suku kata demi suku kata dipisah tanda hubung, dengan ejaan bahasa Indonesia
(contoh gaya penulisan: "se-e-kor", "si-kor", "ke-e-nam", "ke-nam").
Jangan membetulkan dan jangan menebak dari tulisannya. Kalau ada suku kata yang hilang atau bunyinya aneh, tulis apa adanya.
Lalu beri label: "benar" (dilafalkan baku) atau "salah".
Jawab JSON saja: {{"KATA": {{"bunyi": "...", "label": "..."}}}}"""


def ask(path: str, token: str, prompt: str = PROMPT) -> dict:
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(open(path, "rb").read()).decode()}},
            {"text": prompt.format(tokens=token)},
        ]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        data=json.dumps(body).encode(), headers={"x-goog-api-key": KEY, "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            out = json.load(urllib.request.urlopen(req, timeout=120))
            answer = json.loads(out["candidates"][0]["content"]["parts"][-1]["text"])
            answer = answer.get(token) or next(iter(answer.values()))
            return {"bunyi": str(answer.get("bunyi", "")), "label": str(answer.get("label", ""))}
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 503) or attempt == 5:
                raise
        except (KeyError, IndexError, ValueError, StopIteration, AttributeError, OSError):
            if attempt == 5:  # an empty or malformed answer: asked again, then recorded as no answer
                return {"bunyi": "", "label": "error"}
        time.sleep(4 * (attempt + 1))


def phones(bunyi: str) -> str:
    """Gemini's spelling as acro.py phones, so "kyu ris" = "kyuris" and "eich" = "eic"."""
    text = bunyi.lower()
    for a, b in (("tch", "c"), ("ch", "c"), ("sh", "sy"), ("ee", "i"), ("oo", "u")):
        text = text.replace(a, b)
    return "".join(g2p(text))


def judge(path: str, token: str, prompt: str = PROMPT) -> dict:
    asks = [ask(path, token, prompt), ask(path, token, prompt)]
    if phones(asks[0]["bunyi"]) != phones(asks[1]["bunyi"]):
        asks.append(ask(path, token, prompt))
    top, n = Counter(phones(a["bunyi"]) for a in asks).most_common(1)[0]
    majority = next(a["bunyi"] for a in asks if phones(a["bunyi"]) == top) if n >= 2 and top else None
    return {"asks": asks, "majority": majority}


def words(out: str):
    """se-/ke- + e words: right when the heard syllables keep both e's ("se-e-kor"), majority of the asks."""
    rows = [f for f in json.load(open(out + "/manifest.json"))["files"] if f["set"] == "seword"]
    with ThreadPoolExecutor(int(os.environ.get("JUDGE_THREADS", "8"))) as pool:
        results = list(pool.map(lambda f: judge(f"{out}/{f['file']}", f["word"], WORD_PROMPT), rows))
    done = {f["file"]: {**r, "word": f["word"], "variant": f["variant"],
                        "right": bool(r["majority"]) and "ee" in re.sub(r"[^a-z]", "", r["majority"].lower().replace("é", "e"))}
            for f, r in zip(rows, results)}
    json.dump(done, open(out + "/gemini_words.json", "w"), ensure_ascii=False, indent=0)
    calls = sum(len(r["asks"]) for r in done.values())
    print(f"{len(done)} clips, {calls} calls")
    print("| word | written | right (both e's heard) | heard (majority) |\n|---|---|---|---|")
    for word in dict.fromkeys(r["word"] for r in done.values()):
        for variant in ("plain", "hyphen", "space"):
            rs = [r for r in done.values() if r["word"] == word and r["variant"] == variant]
            heard = Counter((r["majority"] or "tidak pasti") for r in rs).most_common(3)
            print(f"| {word} | {variant} | {sum(r['right'] for r in rs)}/{len(rs)} | " + ", ".join(f"{h} x{n}" for h, n in heard) + " |")


def main():
    if sys.argv[1] == "--words":
        return words(sys.argv[2])
    out, temp = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None
    rows = [r for r in json.load(open(out + "/heard.json")) if temp in (None, r["temp"])]
    done = json.load(open(out + "/gemini.json")) if os.path.exists(out + "/gemini.json") else {}
    todo = [r for r in rows if r["file"] not in done and os.path.exists(f"{out}/{r['file']}")]
    print(f"{len(done)} answered, {len(todo)} to ask", flush=True)
    with ThreadPoolExecutor(int(os.environ.get("JUDGE_THREADS", "8"))) as pool:
        for i, (r, result) in enumerate(zip(todo, pool.map(lambda r: judge(f"{out}/{r['file']}", r["acronym"]), todo)), 1):
            done[r["file"]] = result
            if i % 50 == 0 or i == len(todo):
                json.dump(done, open(out + "/gemini.json", "w"), ensure_ascii=False, indent=0)
                print(i, flush=True)


if __name__ == "__main__":
    main()
