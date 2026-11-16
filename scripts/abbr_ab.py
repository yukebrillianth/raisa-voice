"""A/B of how the same word is written to VoxCPM2 (raw acronym vs spelled out vs spaced letters).

    docker compose run --rm --no-deps -v ./scripts:/scripts -v ./out:/out orchestrator \
      python /scripts/abbr_ab.py

Text goes to the TTS server as-is (no normalizer). Writes out/abbr_ab/<case>_<variant>_<n>.wav
and prints Whisper's transcript of each, so both ear and ASR can judge.
"""

import asyncio
import io
import os
import wave

import aiohttp
import numpy as np

TTS_URL = os.environ.get("TTS_URL", "http://tts:8000")
ASR_URL = os.environ.get("ASR_URL", "http://asr:8000")
OUT = "/out/abbr_ab"
SAMPLES = 3
CASES = {
    "usb": ("Kabel {} ada di laci meja.", ["USB", "yu es bi", "U S B"]),
    "pdf": ("Struknya dikirim dalam bentuk {} ya.", ["PDF", "pidief", "pi di ef", "P D F"]),
    "qris": ("Pembayaran bisa pakai {} di kasir.", ["QRIS", "kyuris", "Q R I S"]),
    "gpu": ("Servernya pakai {} terbaru.", ["GPU", "ji pi yu", "G P U"]),
    "hdmi": ("Sambungkan kabel {} ke layar.", ["HDMI", "ha de em i", "H D M I"]),
    "wib": ("Acaranya mulai pukul sembilan {}.", ["WIB", "we i be", "W I B"]),
    "its": ("Selamat datang di {} Surabaya.", ["ITS", "i te es", "I T S"]),
    "nama": ("Kak {} dari tim robotika sudah menunggu di lobi.", ["ANDI", "Andi"]),
    "himatektro": ("Tim {} sudah tiba di aula.", ["HIMATEKTRO", "Himatektro"]),
    "raisa": ("Saya {}, robot pelayan di sini.", ["RAISA", "Raisa"]),
    "suhu": ("Suhu ruangan hari ini {}.", ["23°C", "dua puluh tiga derajat Celsius"]),
    "rupiah": ("Harga tiketnya {} per orang.", ["Rp 45.000", "empat puluh lima ribu rupiah"]),
}


def to_wav16k(pcm: bytes, rate: int) -> bytes:
    x = np.frombuffer(pcm, "<i2").astype(np.float32)
    step = rate // 16000  # 48 kHz -> 16 kHz; mean of 3 is a crude low-pass, enough for Whisper
    x = x[: len(x) // step * step].reshape(-1, step).mean(axis=1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(x.astype("<i2").tobytes())
    return buf.getvalue()


async def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []
    async with aiohttp.ClientSession() as session:
        for case, (template, variants) in CASES.items():
            for v, word in enumerate(variants):
                text = template.format(word)
                heard = []
                for n in range(SAMPLES):
                    async with session.post(f"{TTS_URL}/v1/tts", json={"text": text}) as r:
                        r.raise_for_status()
                        rate, pcm = int(r.headers["X-Sample-Rate"]), await r.read()
                    with wave.open(f"{OUT}/{case}_{v}_{n}.wav", "wb") as w:
                        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate), w.writeframes(pcm)
                    # Neutral hotword so the default list (ITS, RAISA, ...) does not bias the check.
                    async with session.post(
                        f"{ASR_URL}/v1/transcribe", data=to_wav16k(pcm, rate), params={"hotwords": "halo"}
                    ) as r:
                        heard.append((await r.json())["text"])
                lines.append(f"{case}_{v} | {text}\n" + "".join(f"     {n}: {h}\n" for n, h in enumerate(heard)))
                print(lines[-1], flush=True)
    with open(f"{OUT}/index.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


asyncio.run(main())
