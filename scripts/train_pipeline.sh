#!/usr/bin/env bash
# Sequential RL training: run tasks one after another; stop on first failure.
#
# Usage:
#   ./scripts/train_pipeline.sh
#   ./scripts/train_pipeline.sh --env.scene.num-envs=2048
#   nohup ./scripts/train_pipeline.sh > logs/train_pipeline.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TASKS=(
  #Mini2S-Walk-Flat
  Mini2S-Turn-Flat
  Mini2SW-Turn-Flat
  #Mini2SW-Walk-Flat
  #Lite3-Walk-Flat
  Lite3-Turn-Flat
)

EXTRA_ARGS=("$@")

for task in "${TASKS[@]}"; do
  echo "=== start $task $(date -Iseconds) ==="
  python scripts/train.py "$task" --env.scene.num-envs=4096 "${EXTRA_ARGS[@]}"
  echo "=== done  $task $(date -Iseconds) ==="
done

echo "=== pipeline finished $(date -Iseconds) ==="
