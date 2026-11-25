#!/usr/bin/env bash
# Reproduce the full experiment matrix from Section 5.
# Usage: edit DATA_PATH_* below, then:  bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RADIOML="GOLD_XYZ_OSC.0001_16.hdf5"   # RadioML2018.01a HDF5 from DeepSig
FIVEGRADIO="5GRadio.hdf5"             # released with the TFRN code (X/Y/Z)
SEEDS="0 1 2"
COMMON="--epochs 200 --batch-size 128 --lr 1e-4 --seeds $SEEDS --num-workers 4"

# ---- main comparison: all models on both datasets -------------------------
for ds in radioml2018 5gradio; do
  if [ "$ds" = radioml2018 ]; then DP="$RADIOML"; else DP="$FIVEGRADIO"; fi
  for model in resnet mcnet lstm2 mcldnn feat vtmcnet tfrn; do
    python train.py --dataset "$ds" --data-path "$DP" --model "$model" \
      --exp-name "${ds}_${model}" $COMMON
  done
done

# ---- ablations ------------------------------------------------------------
# single domain
for ds in radioml2018 5gradio; do
  if [ "$ds" = radioml2018 ]; then DP="$RADIOML"; else DP="$FIVEGRADIO"; fi
  for mode in iq tf; do
    python train.py --dataset "$ds" --data-path "$DP" --model tfrn \
      --mode "$mode" --exp-name "${ds}_single_${mode}" $COMMON
  done
done

# fusion: CAD / CST (CCT is the default, already trained above)
for ds in radioml2018 5gradio; do
  if [ "$ds" = radioml2018 ]; then DP="$RADIOML"; else DP="$FIVEGRADIO"; fi
  for fus in cad cst; do
    python train.py --dataset "$ds" --data-path "$DP" --model tfrn \
      --fusion "$fus" --exp-name "${ds}_fusion_${fus}" $COMMON
  done
done

# MSCR depth + No MultiScale (RadioML2018.01a)
for n in 2 4 6 8; do
  python train.py --dataset radioml2018 --data-path "$RADIOML" --model tfrn \
    --num-mscr "$n" --exp-name "radioml2018_mscr_${n}" $COMMON
done
python train.py --dataset radioml2018 --data-path "$RADIOML" --model tfrn \
  --no-multiscale --exp-name radioml2018_mscr_nomulti $COMMON

# Transformer heads + No Trans
for h in 4 16; do
  python train.py --dataset radioml2018 --data-path "$RADIOML" --model tfrn \
    --tf-heads "$h" --exp-name "radioml2018_trans_h${h}" $COMMON
done
python train.py --dataset radioml2018 --data-path "$RADIOML" --model tfrn \
  --no-transformer --exp-name radioml2018_trans_none $COMMON

# ---- summary table with paired t-tests -----------------------------------
python report.py --runs-dir runs --out report.md
