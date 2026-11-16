#!/bin/bash
# LoRA v2 final checks on hybrid_s0280 with only the tts service paused (llm and asr keep running).
#   cwords: how VoxCPM2 reads Indonesian c-words written plain / tj / ch / CMUdict braces (split_cwords.json)
#   sweep:  cfg 2.0/2.5/3.0 x timesteps 10/16 on the product text (speech_text.normalize), quality via
#           v2_gen/v2_ana, then TTFAB/RTF of each config through tts/server.py (v2_ttfab.py; mode "ttfab" = timing only)
#   tmux new -d -s raisa-final 'bash ~/yuke/raisa-voice-train/v2/run_final.sh sweep'
# On any exit the lock goes and the tts service comes back as compose has it.
set -euo pipefail
MODE=$1
Y=$HOME/yuke; T=$Y/raisa-voice-train; S=$Y/raisa-voice; D=$Y/raisa-voice-data600
CK=ckpt_v2_hybrid/step_0000280
exec >>"$T/run_final_$MODE.log" 2>&1
UIDGID="$(id -u):$(id -g)"

restore() {
  rc=$?
  set +e
  echo "== restore (exit $rc) $(date)"
  docker rm -f raisa-v2-eval raisa-v2-srv >/dev/null 2>&1
  rm -f "$S/TRAINING.lock"
  docker run --rm -v "$T":/o alpine chown -R "$UIDGID" /o
  (cd "$S" && docker compose up -d tts)
  for i in $(seq 120); do (cd "$S" && docker compose exec -T tts curl -sf localhost:8000/health) && break; sleep 5; done
  echo
  (cd "$S" && docker compose run --rm --no-deps -v ./scripts:/scripts -v ./out:/out orchestrator \
     python /scripts/try_client.py ws://orchestrator:8765/ws /out/q16.wav --out /out/reply_finalcheck.wav) 2>&1 | tail -4
  echo "== done $(date) (exit $rc)"
}
trap restore EXIT
trap 'exit 130' INT TERM HUP

echo "== start $MODE $(date)"
touch "$S/TRAINING.lock"
(cd "$S" && docker compose stop tts)
sleep 5
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$free" -ge 12000 ] || { echo "only $free MiB free on the GPU, aborting"; exit 1; }

HF="-v $HOME/.cache/huggingface:/hf:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1"
RUN="docker run --rm --name raisa-v2-eval --gpus all --ipc=host $HF -v $Y:/yuke -w /yuke/raisa-voice-train/v2 -e PYTHONPATH=/yuke/raisa-voice-train -e GEN_GPU_UTIL=0.3"

if [ "$MODE" = cwords ]; then
  E=/yuke/raisa-voice-train/eval_cwords; SPLIT="-e SPLIT=/yuke/raisa-voice-data600/split_cwords.json"
  $RUN $SPLIT raisa-voice-train python v2_gen.py $E text hybrid_s0280=/yuke/raisa-voice-train/$CK
  $RUN $SPLIT --entrypoint python raisa-voice-asr v2_ana.py asr $E
  exit 0
fi

E=/yuke/raisa-voice-train/eval_final; SPLIT="-e SPLIT=/yuke/raisa-voice-data600/split_final.json"
CONFIGS="2.0:10 2.5:10 3.0:10 2.0:16 2.5:16 3.0:16"
if [ "$MODE" = sweep ]; then  # "ttfab" only redoes the timing below
# split_v2.json + "final": what the robot now sends to the TTS for each sentence.
docker run --rm -v "$S/orchestrator":/orchestrator:ro -v "$D":/d -w /orchestrator raisa-voice-orchestrator python -c '
import json, re
from speech_text import normalize
s = json.load(open("/d/split_v2.json"))
for x in s["sentences"]:
    x["final"] = " ".join(normalize(p) for p in re.split(r"(?<=[.!?])\s+", x["orig"]))
json.dump(s, open("/d/split_final.json", "w"), ensure_ascii=False, indent=1)'
for c in $CONFIGS; do
  echo "== gen $c $(date)"
  $RUN $SPLIT -e CFG=${c%:*} -e TIMESTEPS=${c#*:} raisa-voice-train python v2_gen.py $E final c${c%:*}_t${c#*:}=/yuke/raisa-voice-train/$CK
done
echo "== measure $(date)"; $RUN $SPLIT raisa-voice-train python v2_ana.py measure $E
echo "== asr $(date)"; $RUN $SPLIT --entrypoint python raisa-voice-asr v2_ana.py asr $E
echo "== report $(date)"; $RUN $SPLIT raisa-voice-train python v2_ana.py report $E /yuke/raisa-voice-train/v2_sweep_report.md -
fi
# tts/server.py keeps nano-vllm's defaults (cfg 2.0, 10 steps); this copy reads them from the env.
sed -e 's/max_num_seqs=8,/inference_timesteps=int(os.environ["TTS_TIMESTEPS"]), max_num_seqs=8,/' \
    -e 's/"temperature": 0.7,/"temperature": 0.7, "cfg_value": float(os.environ["TTS_CFG"]),/' \
    "$S/tts/server.py" > "$T/v2/server_sweep.py"
grep -q TTS_CFG "$T/v2/server_sweep.py" && grep -q TTS_TIMESTEPS "$T/v2/server_sweep.py"
for c in $CONFIGS; do
  echo "== ttfab $c $(date)"
  docker run -d --name raisa-v2-srv --gpus all -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v "$S/voices":/voices:ro -v "$T/$CK":/lora:ro -v "$T/v2/server_sweep.py":/app/server.py:ro \
    -e VOICE_LORA=/lora -e TTS_CFG=${c%:*} -e TTS_TIMESTEPS=${c#*:} raisa-voice-tts >/dev/null
  for i in $(seq 60); do docker exec raisa-v2-srv curl -sf localhost:8000/health >/dev/null && break; sleep 5; done
  docker run --rm --network container:raisa-v2-srv -v "$T/v2":/v2:ro -v "$D":/d:ro raisa-voice-orchestrator \
    python /v2/v2_ttfab.py c${c%:*}_t${c#*:} | tee -a "$T/v2_sweep_ttfab.txt"
  docker rm -f raisa-v2-srv >/dev/null
done
