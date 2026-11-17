#!/bin/bash
# LoRA merge test: is base+LoRA folded into the weights (merge_lora.py) the same voice, and is it faster when served?
# 1. merge_test.py: unmerged vs merged at temperature 0, 10 sentences (stack stopped).
# 2. served A/B/A/B: the tts service unmerged (VOICE_LORA) and merged (VOICE_BASE, VOICE_LORA=none), each time
#    TTS first byte + RTF alone (v2_ttfab.py, 15 sentences) and 5 robots at once (bench_robots.sh, 3 runs).
# On any exit: lock removed, stack back to its defaults.
#   LORA=raisa-lora-v2 MERGED=raisa-v2-merged tmux new -d -s raisa-merge 'bash ~/yuke/raisa-voice-train/v3/run_merge.sh'
set -euo pipefail
Y=$HOME/yuke; T=$Y/raisa-voice-train; S=$Y/raisa-voice; E=$T/eval_v3/merge
LORA=${LORA:-raisa-lora-v2}; MERGED=${MERGED:-raisa-v2-merged}
LOG=$T/run_merge.log
exec >>"$LOG" 2>&1
UIDGID="$(id -u):$(id -g)"

restore() {
  rc=$?
  set +e
  echo "== restore (exit $rc) $(date)"
  docker rm -f raisa-merge-test >/dev/null 2>&1
  rm -f "$S/TRAINING.lock"
  (cd "$S" && docker compose up -d)
  docker run --rm -v "$T/eval_v3":/o alpine chown -R "$UIDGID" /o
  for i in $(seq 120); do (cd "$S" && docker compose exec -T tts curl -sf localhost:8000/health) && break; sleep 5; done
  echo "== done $(date) (exit $rc)"
}
trap restore EXIT
trap 'exit 130' INT TERM HUP

echo "== start $(date) lora=$LORA merged=$MERGED"
mkdir -p "$E"
touch "$S/TRAINING.lock"
(cd "$S" && docker compose stop)
sleep 5
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$free" -ge 24000 ] || { echo "only $free MiB free on the GPU, aborting"; exit 1; }

echo "== equivalence $(date)"
docker run --rm --name raisa-merge-test --gpus all --ipc=host -v $HOME/.cache/huggingface:/hf:ro -e HF_HOME=/hf \
  -e HF_HUB_OFFLINE=1 -v $Y:/yuke -w /yuke/raisa-voice-train/v3 --entrypoint python raisa-voice-tts \
  merge_test.py /yuke/raisa-voice/voices/$LORA /yuke/raisa-voice/voices/$MERGED /yuke/raisa-voice-train/eval_v3/merge

serve() {  # $1 label, then env for compose
  local label=$1; shift
  echo "== serve $label $(date)"
  (cd "$S" && env "$@" docker compose up -d)
  for i in $(seq 120); do (cd "$S" && docker compose exec -T tts curl -sf localhost:8000/health) >/dev/null 2>&1 && break; sleep 5; done
  (cd "$S" && docker compose exec -T tts env | grep VOICE_)
  sleep 20  # orchestrator waits for healthy services
  docker run --rm --network container:raisa-voice-tts-1 -v "$T/v2":/v2:ro -v "$Y/raisa-voice-data600":/d:ro \
    raisa-voice-orchestrator python /v2/v2_ttfab.py "$label" | tee -a "$E/ttfab.txt"
  (cd "$S" && N=5 RUNS=3 bash scripts/bench_robots.sh) | sed "s/^/[$label] /" | tee -a "$E/bench.txt"
}
for round in 1 2; do
  serve "unmerged$round" VOICE_LORA=/voices/$LORA
  serve "merged$round" VOICE_BASE=/voices/$MERGED VOICE_LORA=none
done
echo "== stages done $(date)"
