#!/bin/bash
# LoRA v2 (600 clips): pause the stack, train norm + hybrid, evaluate against v1, write the report and A/B
# wavs. Never deploys: voices/ is untouched. On any exit the lock goes and the stack (LoRA v1) comes back.
#   tmux new -d -s raisa-v2 'bash ~/yuke/raisa-voice-train/v2/run_v2.sh'
set -euo pipefail
Y=$HOME/yuke; T=$Y/raisa-voice-train; S=$Y/raisa-voice; D=$Y/raisa-voice-data600; E=$T/eval_v2
LOG=$T/run_v2.log
exec >>"$LOG" 2>&1
UIDGID="$(id -u):$(id -g)"

restore() {
  rc=$?
  set +e
  echo "== restore (exit $rc) $(date)"
  docker rm -f raisa-lora-v2 raisa-v2-eval >/dev/null 2>&1
  rm -f "$S/TRAINING.lock"
  (cd "$S" && docker compose up -d)
  for d in "$T" "$S/out/lora_v2_ab"; do [ -d "$d" ] && docker run --rm -v "$d":/o alpine chown -R "$UIDGID" /o; done
  for i in $(seq 120); do
    (cd "$S" && docker compose exec -T tts curl -sf localhost:8000/health) && break; sleep 5
  done
  echo
  for i in $(seq 10); do  # the orchestrator starts only after llm/tts/asr are healthy
    sleep 15
    (cd "$S" && docker compose run --rm --no-deps -v ./scripts:/scripts -v ./out:/out orchestrator \
       python /scripts/try_client.py ws://orchestrator:8765/ws /out/q16.wav --out /out/reply_v2check.wav) > /tmp/raisa_v2_try.txt 2>&1 \
      && { tail -15 /tmp/raisa_v2_try.txt; echo "try_client OK"; break; }
    tail -1 /tmp/raisa_v2_try.txt
  done
  echo "== done $(date) (exit $rc)"
}
trap restore EXIT
trap 'exit 130' INT TERM HUP

echo "== start $(date)"
mkdir -p "$E"
touch "$S/TRAINING.lock"
(cd "$S" && docker compose stop)
sleep 5
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$free" -ge 24000 ] || { echo "only $free MiB free on the GPU, aborting"; exit 1; }

HF="-v $HOME/.cache/huggingface:/hf:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1"
: > "$E/times.txt"
for V in norm hybrid; do
  # 565 clips / 16 = 35.3 steps per epoch: 701 iters ~ 20 epochs, a checkpoint every 70 steps ~ 2 epochs.
  sed -e "s#/data/train.jsonl#/data/train_$V.jsonl#" -e "s#/data/holdout.jsonl#/data/holdout_$V.jsonl#" \
      -e "s#^num_iters:.*#num_iters: 701#" -e "s#^max_steps:.*#max_steps: 701#" \
      -e "s#^valid_interval:.*#valid_interval: 70#" -e "s#^save_interval:.*#save_interval: 70#" \
      "$T/lora.yaml" > "$T/v2/lora_v2_$V.yaml"
  mkdir -p "$T/ckpt_v2_$V"
  t0=$(date +%s); echo "== train $V $(date)"
  docker run --rm --name raisa-lora-v2 --gpus all --ipc=host $HF -e HF_DATASETS_CACHE=/tmp/ds \
    -v "$D":/data:ro -v "$T/ckpt_v2_$V":/ckpt -v "$T/v2/lora_v2_$V.yaml":/lora.yaml:ro \
    raisa-voice-train python scripts/train_voxcpm_finetune.py --config_path /lora.yaml
  echo "$V: $(( $(date +%s) - t0 )) s" | tee -a "$E/times.txt"
done

ck() { for s in $(seq 70 70 700); do printf " %s_s%04d=/yuke/raisa-voice-train/ckpt_v2_%s/step_%07d" "$1" "$s" "$1" "$s"; done; }
RUN="docker run --rm --name raisa-v2-eval --gpus all --ipc=host $HF -v $Y:/yuke -w /yuke/raisa-voice-train/v2 -e PYTHONPATH=/yuke/raisa-voice-train"
echo "== generate $(date)"
$RUN raisa-voice-train python v2_gen.py /yuke/raisa-voice-train/eval_v2 norm v1=/yuke/raisa-voice/voices/raisa-lora $(ck norm)
$RUN raisa-voice-train python v2_gen.py /yuke/raisa-voice-train/eval_v2 hybrid $(ck hybrid)
echo "== measure $(date)"
$RUN raisa-voice-train python v2_ana.py measure /yuke/raisa-voice-train/eval_v2
echo "== asr $(date)"
$RUN --entrypoint python raisa-voice-asr v2_ana.py asr /yuke/raisa-voice-train/eval_v2
echo "== report $(date)"
$RUN raisa-voice-train python v2_ana.py report /yuke/raisa-voice-train/eval_v2 /yuke/raisa-voice-train/v2_report.md /yuke/raisa-voice/out/lora_v2_ab
