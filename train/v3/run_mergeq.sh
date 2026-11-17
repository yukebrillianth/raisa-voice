#!/bin/bash
# Merged vs unmerged voice quality, as a sample-level match is not possible (two runs of the SAME weights with the same
# seed already differ: merge_test.py). The merged checkpoint renders the same holdout and acronym sets as eval_v3 did
# for the served LoRA (v3_gen.py, served synth), then the same measures: Whisper CER, tempo, onset, f0, fade, P judge.
#   LORA=raisa-lora-v2 MERGED=raisa-v2-merged NAME=v2 tmux new -d -s raisa-mergeq 'bash ~/yuke/raisa-voice-train/v3/run_mergeq.sh'
set -euo pipefail
Y=$HOME/yuke; T=$Y/raisa-voice-train; S=$Y/raisa-voice; Q=$T/eval_v3/mergeq
LORA=${LORA:-raisa-lora-v2}; MERGED=${MERGED:-raisa-v2-merged}; NAME=${NAME:-v2}
LOG=$T/run_mergeq.log
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

echo "== start $(date) lora=$LORA merged=$MERGED"
touch "$S/TRAINING.lock"
(cd "$S" && docker compose stop)
sleep 5
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$free" -ge 24000 ] || { echo "only $free MiB free on the GPU, aborting"; exit 1; }
HF="-v $HOME/.cache/huggingface:/hf:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1"
RUN="docker run --rm --name raisa-v3-eval --gpus all --ipc=host $HF -v $Y:/yuke -e PYTHONPATH=/yuke/raisa-voice-train:/yuke/raisa-voice-train/v3"
M=/yuke/raisa-voice-train/eval_v3/mergeq
rm -rf "$Q"; mkdir -p "$Q/hold" "$Q/acro"
# the unmerged renders of the same sets are reused (same code, same text); the merged ones are new
cp -r "$T/eval_v3/hold/${NAME}h" "$Q/hold/"; cp -r "$T/eval_v3/acro/${NAME}h" "$Q/acro/"
for set in hold acro; do
  echo "== generate $set $(date)"
  $RUN -w /yuke/raisa-voice-train/v3 -e BASE=/yuke/raisa-voice/voices/$MERGED raisa-voice-train \
    python v3_gen.py $M/$set $set ${NAME}m=none
done
echo "== measure $(date)"
$RUN -w /yuke/raisa-voice-train/v2 -e SPLIT=/yuke/raisa-voice-train/data_v3/split_v3eval.json raisa-voice-train python v2_ana.py measure $M/hold
$RUN -w /yuke/raisa-voice-train/v3 raisa-voice-train python v3_ana.py fade $M/hold
$RUN -w /yuke/raisa-voice-train/v2 -e SPLIT=/yuke/raisa-voice-train/data_v3/split_v3eval.json --entrypoint python raisa-voice-asr v2_ana.py asr $M/hold
$RUN -w /yuke/raisa-voice-train/v3 raisa-voice-train python v3_ana.py hold $M/hold $M/hold_report.md
$RUN -w /yuke/raisa-voice-train/v3 raisa-voice-train python v3_ana.py phones $M/acro
$RUN -w /yuke/raisa-voice-train/v3 raisa-voice-train python3 v3_ana.py acro $M/acro $M/acro_report.md
echo "== stages done $(date)"
