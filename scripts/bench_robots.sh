#!/bin/bash
# N robots asking the same question at the same moment: TTFAB per robot, then the orchestrator's own
# stage timings for that run (ASR, LLM first token, first sentence complete, TTS first chunk).
#   N=5 RUNS=3 CLIP=q16.wav bash scripts/bench_robots.sh      # from the repo root on the host
set -euo pipefail
N=${N:-5}; RUNS=${RUNS:-3}; CLIP=${CLIP:-q16.wav}
for run in $(seq "$RUNS"); do
  since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "run $run"
  docker compose run --rm --no-deps -T -v ./scripts:/scripts -v ./out:/out orchestrator sh -c "
    for i in \$(seq $N); do
      python /scripts/try_client.py ws://orchestrator:8765/ws /out/$CLIP --out /out/bench_\$i.wav --wait 9 | grep -E 'TTFAB|first sound' &
    done; wait"
  docker compose logs --no-log-prefix --since "$since" orchestrator 2>&1 | python3 scripts/stage_times.py
done
