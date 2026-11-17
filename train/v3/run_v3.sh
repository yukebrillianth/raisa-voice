#!/bin/bash
# LoRA v3: pause the stack, smoke-test the eval code, train, score every checkpoint on the holdout, render the acronym
# test for v2 + the candidates, run the phoneme judge. Never deploys: voices/ is untouched. On any exit the lock goes
# and the stack comes back.
#   tmux new -d -s raisa-v3 'bash ~/yuke/raisa-voice-train/v3/run_v3.sh'          # all stages
#   STAGES="hold acro" tmux new -d -s raisa-v3 'bash ...'                           # eval again without training
set -euo pipefail
Y=$HOME/yuke; T=$Y/raisa-voice-train; S=$Y/raisa-voice; E=$T/eval_v3; C=$T/ckpt_v3
STAGES=${STAGES:-"smoke train hold acro"}
LOG=$T/run_v3.log
exec >>"$LOG" 2>&1
UIDGID="$(id -u):$(id -g)"

restore() {
  rc=$?
  set +e
  echo "== restore (exit $rc) $(date)"
  docker rm -f raisa-lora-v3 raisa-v3-eval >/dev/null 2>&1
  rm -f "$S/TRAINING.lock"
  (cd "$S" && docker compose up -d)
  for d in "$T"; do docker run --rm -v "$d":/o alpine chown -R "$UIDGID" /o; done
  for i in $(seq 120); do
    (cd "$S" && docker compose exec -T tts curl -sf localhost:8000/health) && break; sleep 5
  done
  echo
  for i in $(seq 10); do  # the orchestrator starts only after llm/tts/asr are healthy
    sleep 15
    (cd "$S" && docker compose run --rm --no-deps -v ./scripts:/scripts -v ./out:/out orchestrator \
       python /scripts/try_client.py ws://orchestrator:8765/ws /out/q16.wav --out /out/reply_v3check.wav) > /tmp/raisa_v3_try.txt 2>&1 \
      && { tail -15 /tmp/raisa_v3_try.txt; echo "try_client OK"; break; }
    tail -1 /tmp/raisa_v3_try.txt
  done
  echo "== done $(date) (exit $rc)"
}
trap restore EXIT
trap 'exit 130' INT TERM HUP

echo "== start $(date) stages: $STAGES"
mkdir -p "$E"
touch "$S/TRAINING.lock"
(cd "$S" && docker compose stop)
sleep 5
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$free" -ge 24000 ] || { echo "only $free MiB free on the GPU, aborting"; exit 1; }

HF="-v $HOME/.cache/huggingface:/hf:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1"
RUN="docker run --rm --name raisa-v3-eval --gpus all --ipc=host $HF -v $Y:/yuke -e PYTHONPATH=/yuke/raisa-voice-train:/yuke/raisa-voice-train/v3"
V2=v2h=/yuke/raisa-voice/voices/raisa-lora-v2
# 1089 clips / 16 = 68 steps per epoch: 15 epochs = 1020 steps, a checkpoint (and validation) every epoch.
SPE=68; STEPS=$((15 * SPE + 1))
ck() { for s in $(seq $SPE $SPE $((STEPS - 1))); do printf " v3_s%04d=/yuke/raisa-voice-train/ckpt_v3/step_%07d" "$s" "$s"; done; }

if [[ $STAGES == *smoke* ]]; then
  echo "== smoke $(date)"
  rm -rf "$T/eval_v3_smoke"
  $RUN -w /yuke/raisa-voice-train/v3 -e LIMIT=4 raisa-voice-train python v3_gen.py /yuke/raisa-voice-train/eval_v3_smoke acro $V2
  $RUN -w /yuke/raisa-voice-train/v3 raisa-voice-train python v3_ana.py phones /yuke/raisa-voice-train/eval_v3_smoke
  ls "$T/eval_v3_smoke/v2h" | head
fi

if [[ $STAGES == *train* ]]; then
  sed -e "s#/data/train.jsonl#/m/train_v3.jsonl#" -e "s#/data/holdout.jsonl#/m/holdout_v3.jsonl#" \
      -e "s#^num_iters:.*#num_iters: $STEPS#" -e "s#^max_steps:.*#max_steps: $STEPS#" \
      -e "s#^valid_interval:.*#valid_interval: $SPE#" -e "s#^save_interval:.*#save_interval: $SPE#" \
      "$T/lora.yaml" > "$T/v3/lora_v3.yaml"
  mkdir -p "$C"
  t0=$(date +%s); echo "== train $(date)"
  docker run --rm --name raisa-lora-v3 --gpus all --ipc=host $HF -e HF_DATASETS_CACHE=/tmp/ds \
    -v "$Y/raisa-voice-data600":/d600:ro -v "$Y/raisa-voice-data-v3":/dv3:ro -v "$T/data_v3":/m:ro \
    -v "$C":/ckpt -v "$T/v3/lora_v3.yaml":/lora.yaml:ro \
    raisa-voice-train python scripts/train_voxcpm_finetune.py --config_path /lora.yaml
  echo "train: $(( $(date +%s) - t0 )) s" | tee "$E/times.txt"
  nvidia-smi --query-gpu=memory.used --format=csv
fi

if [[ $STAGES == *hold* ]]; then
  echo "== hold generate $(date)"
  rm -rf "$E/hold"; mkdir -p "$E/hold"
  $RUN -w /yuke/raisa-voice-train/v3 raisa-voice-train python v3_gen.py /yuke/raisa-voice-train/eval_v3/hold hold $V2 $(ck)
  echo "== hold measure $(date)"
  $RUN -w /yuke/raisa-voice-train/v2 -e SPLIT=/yuke/raisa-voice-train/data_v3/split_v3eval.json raisa-voice-train python v2_ana.py measure /yuke/raisa-voice-train/eval_v3/hold
  $RUN -w /yuke/raisa-voice-train/v3 raisa-voice-train python v3_ana.py fade /yuke/raisa-voice-train/eval_v3/hold
  echo "== hold asr $(date)"
  $RUN -w /yuke/raisa-voice-train/v2 -e SPLIT=/yuke/raisa-voice-train/data_v3/split_v3eval.json --entrypoint python raisa-voice-asr v2_ana.py asr /yuke/raisa-voice-train/eval_v3/hold
  $RUN -w /yuke/raisa-voice-train/v3 raisa-voice-train python v3_ana.py hold /yuke/raisa-voice-train/eval_v3/hold /yuke/raisa-voice-train/v3/hold_report.md
fi

if [[ $STAGES == *acro* ]]; then
  echo "== acro generate $(date)"
  models=""
  for m in $(cat "$E/hold/candidates.txt"); do models+=" $m=/yuke/raisa-voice-train/ckpt_v3/step_000${m#v3_s}"; done
  rm -rf "$E/acro"; mkdir -p "$E/acro"
  $RUN -w /yuke/raisa-voice-train/v3 raisa-voice-train python v3_gen.py /yuke/raisa-voice-train/eval_v3/acro acro $V2 $models
  echo "== acro phones $(date)"
  $RUN -w /yuke/raisa-voice-train/v3 raisa-voice-train python v3_ana.py phones /yuke/raisa-voice-train/eval_v3/acro
fi
echo "== stages done $(date)"
