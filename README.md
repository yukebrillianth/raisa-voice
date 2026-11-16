# raisa-voice

Local replacement for Gemini Live on the RTX 5090 host: the robot gateway streams
mic audio over one WebSocket and gets spoken answers and tool calls back. Sized for
5 robots talking at once.

```
robot 16 kHz PCM ──ws──> orchestrator (Pipecat, one pipeline per robot)
                           Silero VAD + Smart Turn v3.2   end of turn
                           Whisper large-v3-turbo (asr)   one request per speech segment
                           Gemma 4 12B (llm)              answer + tool calls, old turns summarized
                           VoxCPM2 on nano-vllm (tts)     one request per sentence, batched across robots
robot 24 kHz PCM <──ws──
```

| service | what | port |
| --- | --- | --- |
| `llm` | llama.cpp server, Gemma 4 12B QAT Q4, 5 slots x ~13k tokens, thinking off | internal 8080 |
| `asr` | faster-whisper large-v3-turbo, shared by all sessions, Silero VAD filter against hallucinations | internal 8000 |
| `tts` | VoxCPM2 on nano-vllm-voxcpm (batches concurrent requests), voice from LoRA v2 (`voices/raisa-lora-v2/`, see `train/`) plus `voices/raisa-ref.wav` | internal 8000 |
| `orchestrator` | Pipecat pipeline per connection, `ws://<host>:8765/ws`, web simulator at `/` | 8765 |

## Run

The voice files are not in git (private HF repo). Fetch them into `voices/` first:

```bash
hf download yukebrillianth/raisa-voxcpm2-lora v2/lora_weights.safetensors v2/lora_config.json --local-dir /tmp/raisa
mkdir -p voices/raisa-lora-v2 && mv /tmp/raisa/v2/* voices/raisa-lora-v2/
hf download yukebrillianth/raisa-voxcpm2-lora raisa-ref.wav raisa-ref.txt --local-dir voices
```

```bash
cp .env.example .env && sed -i "s/change-me/$(openssl rand -hex 24)/" .env   # VOICE_TOKEN
docker compose up -d --build          # first start downloads VoxCPM2 (~5 GB) and Whisper turbo (~1.6 GB)
docker compose logs -f orchestrator   # turn, STT, LLM and TTS timings are logged per turn
```

Needs the Gemma GGUF already in `~/.cache/huggingface/hub` (see `compose.yaml`) and
about 23 GB of free VRAM (llm 11 GB, tts 8.4 GB, asr 2.8 GB). The voice is
LoRA v2 in `voices/raisa-lora-v2/` (the voice actor's 600 clips, picked by measured speech in
`train/README.md`; v1 stays in `voices/raisa-lora/`, set `VOICE_LORA` to go back) with `voices/raisa-ref.wav` plus its transcript `voices/raisa-ref.txt` as continuation prompt and
reference. All checkpoints and the training data are in the private HF repos
`yukebrillianth/raisa-voxcpm2-lora` and `yukebrillianth/raisa-voice-dataset`.

## Simulator (from a laptop)

The page needs a secure context for the microphone, which `localhost` is, so tunnel the port:

```bash
ssh -N -L 8765:localhost:8765 teaching-factory@10.7.101.217
```

Then open http://localhost:8765 (not http://10.7.101.217:8765: plain http there blocks the mic, the page then
falls back to text mode and says so) in Chrome or Safari, paste `VOICE_TOKEN` from the
server's `.env`, and press Connect. Use a headset: without the robot's hardware AEC,
the speaker leaks into the mic and triggers barge-in. Mock tools `antar_ke_titik` and
`cari_info` answer automatically.

## Protocol

Binary frames are raw PCM16LE mono (16 kHz up, 24 kHz down). JSON messages are listed
in `orchestrator/protocol.py`; they mirror Gemini Live (`setup_complete`, `tool_call`,
`interrupted`, `turn_complete`) so the gateway adapter stays thin. The client sends
`setup` with `token` first and starts streaming audio after `setup_complete`.
`clear_context` starts a fresh conversation for the next visitor.

## Test

```bash
docker compose exec orchestrator python -m pytest -q tests
docker compose run --rm --no-deps -v ./scripts:/scripts -v ./out:/out orchestrator \
  python /scripts/try_client.py ws://orchestrator:8765/ws /out/q16.wav --barge-in /out/b16.wav
```

`try_client.py` plays the robot: streams a 16 kHz WAV in real time, answers tool calls
with a dummy result, and prints TTFAB, audio gaps and barge-in timing. It reads the
token from `VOICE_TOKEN` (set inside the orchestrator container).
