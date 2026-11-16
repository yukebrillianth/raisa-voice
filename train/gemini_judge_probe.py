"""Ask Gemini how an acronym was actually pronounced in a wav. Usage: gemini_judge.py MODEL"""
import base64, json, os, sys, time, urllib.error, urllib.request

MODEL = sys.argv[1]
KEY = os.environ["GEMINI_API_KEY"]
R = "/Users/yukebrillianth/IRIS/iohits-tts-v1/raisa-voice/out/"
# (file, tokens to judge, what the user heard)
CASES = [
    ("lora_v2_final/02_wib_final.wav", ["WIB"], "SALAH: wi ai bi"),
    ("lora_v2_final/05_gpu_gb_final.wav", ["GPU", "GB"], "SALAH: gepiu, gibi"),
    ("lora_v2_final/06_hdmi_its_final.wav", ["HDMI", "ITS"], "SALAH: hasdmi"),
    ("lora_v2_ab/u9_gpu_gb_hybrid.wav", ["GPU", "GB"], "benar (Inggris)"),
    ("lora_v2_ab/u10_hdmi_its_hybrid.wav", ["HDMI", "ITS"], "benar (Indonesia)"),
    ("lora_v2_ab/u7_pdf_wib_norm.wav", ["PDF", "WIB"], "WIB benar: we i be"),
    ("lora_v2_ab/h29_dewi_wib_norm.wav", ["WIB"], "benar: we i be"),
    ("lora_v2_ab/u8_usb_tv_hybrid.wav", ["USB", "TV"], "benar (Inggris)"),
    ("lora_v2_ab/u0_qris_v1.wav", ["QRIS"], "benar: kyuris"),
    ("lora_v2_final/02_wib_v1.wav", ["WIB"], "(tidak dikeluhkan)"),
]
PROMPT = """Ini rekaman satu kalimat dari mesin TTS. Tugasmu menilai PELAFALAN, bukan isi.
Untuk tiap singkatan berikut: {tokens}
tulis bunyi yang BENAR-BENAR terdengar, suku kata demi suku kata, dengan ejaan bahasa Indonesia
(contoh gaya penulisan: "ji pi yu", "ge pe u", "gepiu", "ha de em i", "eich di em ai").
Jangan membetulkan dan jangan menebak dari tulisannya. Kalau bunyinya aneh atau ada huruf yang hilang, tulis apa adanya.
Lalu beri label: "inggris" (nama huruf Inggris), "indonesia" (nama huruf Indonesia), "kata" (dibaca sebagai satu kata), atau "kacau".
Jawab JSON saja: {{"SINGKATAN": {{"bunyi": "...", "label": "..."}}}}"""


def ask(path, tokens):
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(open(R + path, "rb").read()).decode()}},
            {"text": PROMPT.format(tokens=", ".join(tokens))},
        ]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        data=json.dumps(body).encode(), headers={"x-goog-api-key": KEY, "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            out = json.load(urllib.request.urlopen(req, timeout=120)); break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 503) or attempt == 5: raise
            time.sleep(8 * (attempt + 1))
    return json.loads(out["candidates"][0]["content"]["parts"][-1]["text"])


for path, tokens, heard in CASES:
    try:
        answers = [ask(path, tokens) for _ in range(3)]  # three times: is the judge stable?
    except Exception as e:
        print(path, "ERROR", str(e)[:200]); continue
    print(f"{path.split('/')[-1]:28s} user: {heard}")
    for t in tokens:
        print(f"    {t:5s} " + "  |  ".join(f"{a.get(t, {}).get('bunyi')} [{a.get(t, {}).get('label')}]" for a in answers))
