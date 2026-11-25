#!/usr/bin/env bash
# 快速试跑（CPU 可承受）：从 RadioML2018.01a 中按类别+SNR 均匀抽 6000 帧，
# 单种子、2 epoch，验证整条流水线可跑通。约 1~3 分钟（取决于 CPU）。
# Usage: bash scripts/quick_demo.sh [path/to/GOLD_XYZ_OSC.0001_16.hdf5]
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${1:-GOLD_XYZ_OSC.0001_16.hdf5}"
PYTHON="${PYTHON:-python}"     # e.g. PYTHON=/path/to/python3.11 bash quick_demo.sh

"$PYTHON" train.py --dataset radioml2018 --data-path "$DATA" --model tfrn \
  --max-samples 6000 --epochs 2 --seeds 0 --batch-size 64 \
  --num-workers 2 --early-stop-patience 2 --exp-name quick_tfrn

"$PYTHON" report.py --runs-dir runs --out quick_report.md
