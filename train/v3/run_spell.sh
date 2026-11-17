#!/bin/bash
# é-hyphen spelling test: raw acronyms vs Indonesian letter names "wé-i-bé" (+ seekor/keenam spellings), 8 samples each,
# on the served merged v2 and on v3 step 612, then the phoneme judge. Stack paused, restored on any exit.
#   tmux new -d -s raisa-spell 'bash ~/yuke/raisa-voice-train/v3/run_spell.sh'
set -euo pipefail
Y=$HOME/yuke; T=$Y/raisa-voice-train; S=$Y/raisa-voice; E=$T/eval_v3/spell
LOG=$T/run_spell.log
exec >>"$LOG" 2>&1
UIDGID="$(id -u):$(id -g)"
restore() {
  rc=$?
  set +e
  echo "== restore (exit $rc) $(date)"
  docker rm -f raisa-v3-eval >/dev/null 2>&1
  rm -f "$S/TRAINING.lock"
  (cd "$S" && docker compose up -d)
  docker run --rm -v "$T/eval_v3":/o alpine chown -R "$UIDGID" /o
  for i in $(seq 120); do (cd "$S" && docker compose exec -T tts curl -sf localhost:8000/health) && break; sleep 5; done
  echo "== done $(date) (exit $rc)"
}
trap restore EXIT
trap 'exit 130' INT TERM HUP
echo "== start $(date)"
touch "$S/TRAINING.lock"
(cd "$S" && docker compose stop)
sleep 5
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$free" -ge 24000 ] || { echo "only $free MiB free on the GPU, aborting"; exit 1; }
HF="-v $HOME/.cache/huggingface:/hf:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1"
RUN="docker run --rm --name raisa-v3-eval --gpus all --ipc=host $HF -v $Y:/yuke -w /yuke/raisa-voice-train/v3 -e PYTHONPATH=/yuke/raisa-voice-train:/yuke/raisa-voice-train/v3"
rm -rf "$E"; mkdir -p "$E"
$RUN -e BASE=/yuke/raisa-voice/voices/raisa-v2-merged raisa-voice-train python v3_gen.py /yuke/raisa-voice-train/eval_v3/spell spell v2m=none
$RUN raisa-voice-train python v3_gen.py /yuke/raisa-voice-train/eval_v3/spell spell v3_s0612=/yuke/raisa-voice-train/ckpt_v3/step_0000612
$RUN -e TESTS=spell_tests.json raisa-voice-train python v3_ana.py phones /yuke/raisa-voice-train/eval_v3/spell
echo "== stages done $(date)"
