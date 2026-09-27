#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"; mkdir -p data checkpoints logs models
HOURS="${HOURS:-5}"; POPULATION="${POPULATION:-8}"; MUTATION_RATE="${MUTATION_RATE:-0.03}"; CROSSOVER_RATE="${CROSSOVER_RATE:-0.08}"
COMMON=(--data data/features.npz --raw data/btcusdt_1m.parquet --horizons 1,5,15,60,240,720 --hours "$HOURS" --population "$POPULATION" --mutation-rate "$MUTATION_RATE" --crossover-rate "$CROSSOVER_RATE" --checkpoint-dir checkpoints --model-dir models --resume)
GPU_COUNT=$(command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null | wc -l || true); GPU_COUNT=${GPU_COUNT:-0}
TOPO=$(command -v nvidia-smi >/dev/null && nvidia-smi topo -m 2>/dev/null || true)
if (( GPU_COUNT > 1 )) && grep -Eq 'NV[1-9]' <<<"$TOPO"; then
  echo "NVLink detected: DDP on $GPU_COUNT GPUs" | tee logs/launcher.log
  torchrun --standalone --nproc_per_node="$GPU_COUNT" -m src.train "${COMMON[@]}" 2>&1 | tee -a logs/ddp.log
elif (( GPU_COUNT > 0 )); then
  echo "No NVLink: $GPU_COUNT independent evolutionary islands" | tee logs/launcher.log
  pids=(); for ((i=0;i<GPU_COUNT;i++)); do
    CUDA_VISIBLE_DEVICES=$i python -m src.train "${COMMON[@]}" --island-id "$i" --population "$(( (POPULATION+GPU_COUNT-1)/GPU_COUNT ))" >"logs/island_$i.log" 2>&1 & pids+=("$!")
  done
  trap 'kill ${pids[*]} 2>/dev/null || true' INT TERM; for p in "${pids[@]}"; do wait "$p"; done; python -m src.select_best
else
  echo "No GPU: CPU multiprocessing/data-loading mode" | tee logs/launcher.log
  python -m src.train "${COMMON[@]}" 2>&1 | tee -a logs/cpu.log
fi
