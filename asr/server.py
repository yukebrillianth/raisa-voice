"""Whisper large-v3-turbo ASR server: POST /v1/transcribe (16 kHz mono PCM16 WAV body) -> {"text"}.

One model on the GPU, shared by every robot session. Whisper invents text on
silence and noise ("Terima kasih." on 2 s of silence), and turbo reports
no_speech_prob 0.0 even then, so non-speech is cut by faster-whisper's Silero VAD
filter instead. Repetition loops and known subtitle-credit phrases are dropped too.
"""

import io
import os
import re
import time
import wave

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from faster_whisper import WhisperModel
from starlette.concurrency import run_in_threadpool

# Whisper picks the language itself, after its VAD filter. Forcing one (or detecting it
# separately on the raw clip, silence included) made it translate a visitor's English into
# Indonesian. Languages outside this set are retried as Indonesian.
LANGUAGES = ("id", "en", "ms", "jw")
# Words Whisper does not know. Without them "What is Teaching Factory ITS?" came out as
# "Apa yang mengajar factory ITS?"; they also steer the language guess.
# Default when the session sends none; the gateway sends the agent's transcription.customVocabulary.
HOTWORDS = "RAISA, ITS, Teaching Factory, Pusat Robotika, FTEIC, NABBIT, IRIS, MAGE, BIMA, lobi, ruang rapat, lab robotika"
# Debug only: keep every request's audio here to study misheard turns. It stores visitors'
# voices, so it stays unset in normal use.
DEBUG_DIR = os.environ.get("ASR_DEBUG_DIR")
# Whisper's own threshold for a repetition loop.
COMPRESSION_RATIO = 2.4
# Whole-segment matches only: a visitor saying "terima kasih" is real speech.
HALLUCINATIONS = {
    "terima kasih telah menonton",
    "terima kasih sudah menonton",
    "sampai jumpa di video berikutnya",
    "jangan lupa like dan subscribe",
    "thank you for watching",
    "thanks for watching",
}

# num_workers=2: two sessions can transcribe at the same time.
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16", num_workers=2)
app = FastAPI()


def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def keep(segment) -> bool:
    return segment.compression_ratio <= COMPRESSION_RATIO and normalize(segment.text) not in HALLUCINATIONS


def transcribe(audio, hotwords: str = HOTWORDS) -> str:
    # Threshold 0.35, not the default 0.5: at 0.5 one-word turns ("Halo.", "Ya.", "Oke.") were
    # dropped as non-speech, while silence, noise and breath stay empty down to 0.2.
    options = dict(beam_size=1, condition_on_previous_text=False, hotwords=hotwords,
                   vad_filter=True, vad_parameters={"threshold": 0.35})
    segments, info = model.transcribe(audio, **options)
    if info.language not in LANGUAGES:  # segments are lazy, so nothing was decoded yet
        segments, info = model.transcribe(audio, language="id", **options)
    text = " ".join(s.text.strip() for s in segments if keep(s)).strip()
    print(f"lang {info.language} {info.language_probability:.2f} text {text!r}", flush=True)
    return text


# Warmup both the VAD and the decoder (VAD alone drops silence before Whisper runs);
# a cold first request took about 1.5 s.
transcribe(np.zeros(16000, dtype=np.float32))
list(model.transcribe(np.zeros(16000, dtype=np.float32), language="id", beam_size=1)[0])


@app.get("/health")
def health():
    return {"ok": True}


def read_wav(body: bytes) -> np.ndarray:
    # Read directly instead of through PyAV: Pipecat always sends this exact format.
    try:
        with wave.open(io.BytesIO(body)) as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
                raise HTTPException(400, "expected 16 kHz mono PCM16 WAV")
            pcm = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as e:
        raise HTTPException(400, f"invalid WAV: {e}")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


@app.post("/v1/transcribe")
async def transcribe_endpoint(request: Request):
    body = await request.body()
    audio = read_wav(body)
    if DEBUG_DIR:
        path = f"{DEBUG_DIR}/{time.strftime('%H%M%S')}_{time.time_ns() % 1000000}.wav"
        open(path, "wb").write(body)
        print(f"saved {path}", flush=True)
    hotwords = request.query_params.get("hotwords") or HOTWORDS
    return {"text": await run_in_threadpool(transcribe, audio, hotwords)}
