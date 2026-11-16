"""Per-robot stage timings of the first answer, from the orchestrator's debug log on stdin.

    docker compose logs --no-log-prefix --since 1m orchestrator | python3 scripts/stage_times.py
Columns (ms): asr = Whisper request, llm = request to first token, sent = first token to first
complete sentence, tts = TTS request to first audio chunk, lead = silence at the start of that audio.
"""
import re
import sys
from datetime import datetime

PATTERNS = {
    "asr": r"WhisperServerSTTService#(\d+) processing time: ([\d.]+)s",
    "llm": r"OpenAILLMService#(\d+) TTFB: ([\d.]+)s",
    "tts": r"VoxCPMTTSService#(\d+) TTFB: ([\d.]+)s",
    "lead": r"VoxCPMTTSService#(\d+) TTFA: [\d.]+s \(([\d.]+)s leading silence",
}
MARKS = {"token": r"OpenAILLMService#(\d+) TTFB: ", "say": r"VoxCPMTTSService#(\d+): Generating TTS"}
rows: dict[str, dict] = {}
for line in sys.stdin:
    for key, pattern in PATTERNS.items():
        if (m := re.search(pattern, line)) and float(m.group(2)) > 0:
            rows.setdefault(m.group(1), {}).setdefault(key, float(m.group(2)) * 1000)
    for key, pattern in MARKS.items():
        if m := re.search(pattern, line):
            at = datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S.%f")
            rows.setdefault(m.group(1), {}).setdefault(key, at)
print("robot   asr   llm  sent   tts  lead")
for robot, r in sorted(rows.items(), key=lambda kv: int(kv[0])):
    if "tts" not in r:
        continue
    sent = (r["say"] - r["token"]).total_seconds() * 1000 if "say" in r and "token" in r else float("nan")
    print(f"#{robot:<5}" + "".join(f"{v:6.0f}" for v in (r.get("asr", 0), r.get("llm", 0), sent, r["tts"], r.get("lead", 0))))
