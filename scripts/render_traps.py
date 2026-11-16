"""Renders tricky sentences through the same text transforms as the TTS service, then VoxCPM2.

    docker compose run --rm --no-deps -v ./scripts:/scripts -v ./out:/out orchestrator \
      python /scripts/render_traps.py

Writes out/traps/NN.wav and out/traps/index.txt (written text -> spoken text), so the
normalizer can be checked by ear.
"""

import asyncio
import os
import wave

import aiohttp

from speech_text import SPEECH_TRANSFORMS, normalize

TTS_URL = os.environ.get("TTS_URL", "http://tts:8000")
OUT = "/out/traps"
TRAPS = [
    "Harga tiket masuk Rp150.000 per orang, anak-anak Rp 75.000,- saja.",
    "Pameran UMKM dibuka pukul 08.00 WIB sampai 16.30 WIB.",
    "Silakan ke Gedung FTEIC ITS lantai 3, ruang 305.",
    "Diskon 25% berlaku sampai 17 Agustus 2026.",
    "Jaraknya sekitar 1,5 km atau 20 menit jalan kaki.",
    "Hubungi admin di 0812-3456-7890 atau +62 31 594 7274.",
    "Jangan lupa bawa KTP, kartu mahasiswa, dll.",
    "Kamu bisa download aplikasinya, lalu login pakai email kampus ya.",
    "The meeting room is on the second floor, next to the lift.",
    "**Penting:** jangan lupa membawa *undangan* ya!",
    "Jadwalnya:\n- Senin pukul 08.00\n- Selasa pukul 13.00",
    "| Hari | Jam |\n|---|---|\n| Senin | 08.00 |\nItu jadwal lengkapnya.",
    "Halo! 😊 Selamat datang di Teaching Factory 🤖",
    "Robot ini juara ke-2 lomba tahun 2025 dengan nilai 87,5.",
    "Rapat jam 14:30, sisa anggaran Rp 1,5 juta.",
    "Saya RAISA, robot pelayan dari Pusat Robotika ITS.",
    "Bayar pakai QRIS, struknya dikirim sebagai PDF, dan kabel USB ada di laci.",
    "Suhu ruangan 23°C, konsumsi daya 14,2 kWh, prosesor 4,2 GHz dengan RAM 32 GB.",
    "Tonton siarannya di ITS TV, ruang VIP ada di lantai 2 dekat tim IT.",
    "Kak ANDI dari BANDUNG dan tim HIMATEKTRO sudah tiba pukul 10.00 WITA.",
    "Please connect the HDMI cable, the room is 23°C and the fee is Rp 45,000.",
    "Your GPU server and API are ready, scan the QR code at 10:30.",
]


async def spoken(text: str) -> str:
    for _, transform in SPEECH_TRANSFORMS:
        text = await transform(text, "sentence")
    return normalize(text).strip()


async def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []
    async with aiohttp.ClientSession() as session:
        for i, text in enumerate(TRAPS, 1):
            say = await spoken(text)
            async with session.post(f"{TTS_URL}/v1/tts", json={"text": say}) as response:
                response.raise_for_status()
                rate = int(response.headers["X-Sample-Rate"])
                pcm = await response.read()
            with wave.open(f"{OUT}/{i:02d}.wav", "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate), w.writeframes(pcm)
            written = text.replace("\n", " / ")
            lines.append(f"{i:02d} | {written}\n   -> {say.replace(chr(10), ' / ')}")
            print(lines[-1], f"({len(pcm) / 2 / rate:.1f} s)", flush=True)
    with open(f"{OUT}/index.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


asyncio.run(main())
