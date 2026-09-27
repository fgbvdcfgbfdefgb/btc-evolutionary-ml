# Evolutionary Bitcoin Multi-Horizon Forecaster

A runnable PyTorch project that downloads and caches 365 days of Binance BTCUSDT 1-minute OHLCV, builds leakage-safe features, and evolves Transformer forecasters for log-return horizons **1, 5, 15, 60, 240, and 720 minutes**. This is research software, not financial advice.

## Layout

```
btc_price.ipynb       end-to-end notebook
run_multigpu.sh        automatic NVLink/DDP, GPU-island, or CPU launcher
src/data.py            cached download, features, window dataset
src/model.py           Transformer forecaster
src/evolution.py       selection, crossover, mutation
src/evaluate.py        RMSE, MAE, directional accuracy, fitness
src/hardware.py        GPU/VRAM/CUDA/NVLink/CPU/RAM/disk detection
src/train.py           CLI, AMP, DDP, checkpoint/resume
src/select_best.py     selects best non-NVLink island
data/ checkpoints/ logs/ models/
```

## Install and run

Python 3.10+ and a current NVIDIA driver are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_multigpu.sh
```

The default wall-clock budget is 5 hours. Override without editing files:

```bash
HOURS=1 POPULATION=12 MUTATION_RATE=0.02 CROSSOVER_RATE=0.05 ./run_multigpu.sh
```

The first run downloads roughly 525,600 candles through Binance's public API and writes `data/btcusdt_1m.parquet`; later runs reuse it. `data/features.npz` is also cached. Availability and geographic restrictions of Binance apply. Delete either cache to rebuild it. The chronological split is 80% train / 20% validation.

## Compute strategy

* **2+ GPUs with NVLink:** `torchrun` DDP; NCCL synchronizes every candidate across all GPUs.
* **GPU(s), no NVLink:** one independent evolutionary island per GPU runs concurrently, followed by selection into `models/best.pt`. This avoids slow PCIe synchronization and explores candidates in parallel.
* **No GPU:** CPU execution with tuned PyTorch threads and parallel DataLoader workers.

AMP uses BF16 where supported and FP16 otherwise. Loaders use pinned memory, persistent workers and prefetching. Features are loaded through NumPy's memory-map interface where supported. Set `--accumulate N` directly in a CLI run if VRAM requires gradient accumulation.

## Resume and outputs

Auto-resume is on. Re-running the launcher loads `checkpoints/evolution_islandN.pt`, including population models, hyperparameters, optimizer state when present, generation, and Python/NumPy/PyTorch RNG states. Checkpoints are written each generation and at the configured interval. SIGTERM triggers a final save.

* `models/best_islandN.pt`: best model from each island
* `models/best.pt`: overall winner in non-NVLink multi-GPU mode
* `checkpoints/evolution_islandN.pt`: resumable evolutionary state
* `logs/*.log`: launcher/training logs

Direct CLI example:

```bash
python -m src.train --hours 5 --population 8 --mutation-rate .03 \
  --crossover-rate .08 --checkpoint-minutes 10 --accumulate 2 --resume
```

View all options with `python -m src.train --help`. The notebook uses the same modules, so notebook and CLI behavior stay aligned.

## Reproducibility notes

A fixed seed controls initialization and evolution. CUDA kernels can still have small nondeterministic differences because cuDNN benchmarking is intentionally enabled for throughput. Fitness is negative validation RMSE (higher is better); MAE and directional accuracy are also recorded. Mutation and crossover defaults are intentionally conservative for stable evolution.
