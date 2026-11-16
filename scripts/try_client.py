"""Test client that plays the robot's role against the orchestrator.

Streams a 16 kHz mono PCM16 WAV in real time, then silence, prints every event
with a timestamp, answers tool calls with a dummy result, and saves the reply.

    python try_client.py ws://localhost:8765/ws question.wav --out reply.wav
    python try_client.py ws://localhost:8765/ws question.wav --barge-in other.wav

TTFAB is measured from the last byte of the user's speech to the first reply byte, "first sound"
to the first reply audio that is not silence (what the visitor actually waits for).
"""

import argparse
import asyncio
import json
import os
import time
import wave

import websockets

CHUNK_SECS = 0.04
IN_RATE = 16000
OUT_RATE = 24000
TOOLS = [
    {
        "name": "antar_ke_titik",
        "description": "Mengantar pengunjung secara fisik ke sebuah lokasi di gedung.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"lokasi": {"type": "STRING", "description": "Nama lokasi tujuan"}},
            "required": ["lokasi"],
        },
    }
]
SYSTEM = (
    "Kamu RAISA, robot pelayan di Teaching Factory Pusat Robotika ITS. "
    "Lokasi yang bisa kamu tuju: lobi, ruang rapat, lab robotika, toilet."
)


def read_pcm(path: str) -> bytes:
    with wave.open(path) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (IN_RATE, 1, 2):
            raise SystemExit(f"{path} must be 16 kHz mono PCM16")
        return w.readframes(w.getnframes())


async def stream(ws, pcm: bytes, trailing_silence_secs: float) -> float:
    """Send audio at real-time pace; return the monotonic time speech ended."""
    step = int(IN_RATE * CHUNK_SECS) * 2
    for i in range(0, len(pcm), step):
        await ws.send(pcm[i : i + step])
        await asyncio.sleep(CHUNK_SECS)
    speech_end = time.monotonic()
    silence = b"\x00" * step
    for _ in range(int(trailing_silence_secs / CHUNK_SECS)):
        await ws.send(silence)
        await asyncio.sleep(CHUNK_SECS)
    return speech_end


async def main(args):
    t0 = time.monotonic()
    log = lambda msg: print(f"[{time.monotonic() - t0:7.3f}] {msg}", flush=True)  # noqa: E731
    reply = bytearray()
    audio_times: list[float] = []
    sound_times: list[float] = []  # audio messages that are not silence
    events: list[tuple[float, str]] = []
    spoken: list[str] = []

    async with websockets.connect(args.url, max_size=None) as ws:
        setup = {"type": "setup", "token": args.token, "systemInstruction": SYSTEM, "tools": TOOLS}
        await ws.send(json.dumps(setup))

        async def answer_tool_calls(event):
            await asyncio.sleep(args.tool_delay)
            for call in event["functionCalls"]:
                response = {"status": "berangkat", "lokasi": call["args"].get("lokasi")}
                result = {"id": call["id"], "name": call["name"], "response": response}
                await ws.send(json.dumps({"type": "tool_response", "functionResponses": [result]}))
                log(f"tool_response sent for {call['name']}")

        async def receive():
            async for message in ws:
                now = time.monotonic()
                if isinstance(message, bytes):
                    audio_times.append(now)
                    if max(abs(v) for v in memoryview(message).cast("h")) > 300:
                        sound_times.append(now)
                    reply.extend(message)
                    continue
                event = json.loads(message)
                events.append((now, event["type"]))
                if event["type"] == "assistant_text":  # one per word: collected, printed at the end
                    spoken.append(event["text"])
                else:
                    log(f"event {event}")
                if event["type"] == "tool_call":
                    # A task, so a slow tool does not stop audio from being received meanwhile.
                    asyncio.create_task(answer_tool_calls(event))

        def first_audio_after(t: float) -> float | None:
            return next((a for a in audio_times if a > t), None)

        receiver = asyncio.create_task(receive())
        while not any(kind == "setup_complete" for _, kind in events):
            if any(kind == "error" for _, kind in events) or receiver.done():
                raise SystemExit("setup rejected")
            await asyncio.sleep(0.01)
        # A real mic streams before anyone talks; VAD needs that lead-in to catch the first word.
        await stream(ws, b"", trailing_silence_secs=0.5)
        speech_end = await stream(ws, read_pcm(args.wav), trailing_silence_secs=0.0)
        log("user speech ended")
        background = asyncio.create_task(stream(ws, b"", trailing_silence_secs=args.wait + 30))

        if args.barge_in:
            while first_audio_after(speech_end) is None:
                await asyncio.sleep(0.01)
            await asyncio.sleep(1.0)
            background.cancel()
            barge_start = time.monotonic()
            log("barge-in speech starts")
            barge_end = await stream(ws, read_pcm(args.barge_in), trailing_silence_secs=0.0)
            background = asyncio.create_task(stream(ws, b"", trailing_silence_secs=args.wait + 30))

        await asyncio.sleep(args.wait)
        background.cancel()
        receiver.cancel()

    # The server paces audio in real time, so a gap in arrival is silence the visitor hears.
    for prev, cur in zip(audio_times, audio_times[1:]):
        if cur - prev > 0.3:
            print(f"audio gap {cur - prev:.2f} s at {prev - t0:.2f} s")
    first = first_audio_after(speech_end)
    print(f"TTFAB turn 1: {(first - speech_end) * 1000:.0f} ms" if first else "TTFAB turn 1: no audio")
    sound = next((a for a in sound_times if a > speech_end), None)
    print(f"first sound turn 1: {(sound - speech_end) * 1000:.0f} ms" if sound else "first sound turn 1: none")
    if args.barge_in:
        cut = next((t for t, kind in events if kind == "interrupted" and t > barge_start), None)
        if cut is None:
            print("barge-in: no interrupted event")
        else:
            # Audio still arriving after the cut (before the barge-in reply) is what the robot must flush.
            stale = [a for a in audio_times if cut < a < (first_audio_after(barge_end) or float("inf"))]
            print(f"barge-in: interrupted {(cut - barge_start) * 1000:.0f} ms after barge-in speech started")
            print(f"barge-in: {len(stale)} stale audio frames after interrupted")
            # Words of the cut answer must not follow it either (the next answer starts after a transcript).
            reply_at = next((t for t, kind in events if kind == "user_transcript" and t > cut), float("inf"))
            late = [t for t, kind in events if kind == "assistant_text" and cut < t < reply_at]
            print(f"barge-in: {len(late)} words of the cut answer after interrupted")
        second = first_audio_after(barge_end)
        print(f"TTFAB turn 2: {(second - barge_end) * 1000:.0f} ms" if second else "TTFAB turn 2: no audio")

    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(OUT_RATE)
        w.writeframes(bytes(reply))
    print(f"robot said ({len(spoken)} words): {' '.join(spoken)}")
    print(f"saved {len(reply) / 2 / OUT_RATE:.2f} s of reply audio to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("wav", help="16 kHz mono PCM16 WAV with the user's question")
    parser.add_argument("--barge-in", help="second WAV spoken 1 s into the reply")
    parser.add_argument("--out", default="reply.wav")
    parser.add_argument("--tool-delay", type=float, default=0.0, help="seconds the dummy tool takes")
    parser.add_argument("--token", default=os.environ.get("VOICE_TOKEN", ""))
    parser.add_argument("--wait", type=float, default=12.0, help="seconds to keep listening")
    asyncio.run(main(parser.parse_args()))
