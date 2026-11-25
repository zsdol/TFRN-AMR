# RTX 4090 云服务器部署指南

## 1. 租机器时怎么选

| 项目 | 建议 | 原因 |
|---|---|---|
| GPU | RTX 4090 24 GB × 1 | 显存远超本模型需求（batch 128 约 2–3 GB），单卡足够 |
| 系统内存 | **≥ 64 GB**（最低 32 GB） | 全量 RadioML2018 的 IQ 数组约 21 GB，加上 Python/HDF5 开销，32 GB 很紧 |
| CPU | ≥ 8 核 | DataLoader worker 做 STFT，8 核以上才喂得饱 4090 |
| 磁盘 | ≥ 100 GB | 数据集 20 GB + 代码 + checkpoint |
| 系统 | Ubuntu 20.04/22.04 | 装 CUDA 版 torch 最省事 |

按国内主流平台（AutoDL / 恒源云 / 滴滴云等）行情，4090 大约 **¥2.5–4/小时**。按下面估算的总时长，全套实验电费约 **¥250–600**（按 2 随机种子）。

## 2. 环境配置（服务器端）

```bash
# Ubuntu 22.04，假设已装好 NVIDIA 驱动 + CUDA 12.1
conda create -n tfrn python=3.11 -y && conda activate tfrn

# CUDA 版 torch（按平台实际 CUDA 版本选 cu121 / cu118）
pip install torch --index-url https://download.pytorch.org/whl/cu121

cd /root/TFRN
pip install -r requirements.txt

# 确认 GPU 可用
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> 注意：不要直接 `pip install -r requirements.txt` 里的 torch（那是 PyPI 通用版），先按上面的 CPU/CUDA 源单独装 torch。

## 3. 上传代码和数据

```bash
# 本地执行（换成你的平台地址）
scp -r TFRN user@host:/root/
scp GOLD_XYZ_OSC.0001_16.hdf5 user@host:/root/TFRN/data/
scp 5GRadio.hdf5 user@host:/root/TFRN/data/
```

## 4. 先验证，再正式跑

```bash
cd /root/TFRN
PYTHON=python bash scripts/quick_demo.sh data/GOLD_XYZ_OSC.0001_16.hdf5   # 约 20~40 秒
python train.py --dataset radioml2018 --data-path data/GOLD_XYZ_OSC.0001_16.hdf5 \
  --model tfrn --device cuda --epochs 2 --seeds 0 --batch-size 256 --max-samples 30000 \
  --exp-name check_gpu
```

确认日志里 `device=cuda`、显存占用正常后再往下走。

## 5. 全套实验（建议用 tmux 挂后台）

```bash
tmux new -s tfrn
# 编辑 scripts/run_all.sh 顶部两个数据文件路径为 data/xxx.hdf5
bash scripts/run_all.sh 2>&1 | tee full_run.log
# Ctrl+B 然后 D 脱离；tmux attach -t tfrn 回来看
```

更稳妥的做法是**分批跑**（便于中途恢复/汇报）：

```bash
# 第 1 批：主实验
for model in resnet mcnet lstm2 mcldnn feat vtmcnet tfrn; do
  python train.py --dataset radioml2018 --data-path data/GOLD_XYZ_OSC.0001_16.hdf5 \
    --model "$model" --device cuda --batch-size 256 --seeds 0 1 \
    --exp-name "radioml2018_$model"
done
# 5GRadio 同理，把 dataset/data-path/exp-name 换掉

# 第 2 批：消融（单域 / 融合 / MSCR / Transformer）
python train.py ... --mode iq  --exp-name radioml2018_single_iq
python train.py ... --mode tf  --exp-name radioml2018_single_tf
python train.py ... --fusion cad --exp-name radioml2018_fusion_cad
python train.py ... --fusion cst --exp-name radioml2018_fusion_cst
python train.py ... --num-mscr 2 --exp-name radioml2018_mscr_2
python train.py ... --num-mscr 4 --exp-name radioml2018_mscr_4
python train.py ... --num-mscr 8 --exp-name radioml2018_mscr_8
python train.py ... --no-multiscale --exp-name radioml2018_mscr_nomulti
python train.py ... --tf-heads 4  --exp-name radioml2018_trans_h4
python train.py ... --tf-heads 16 --exp-name radioml2018_trans_h16
python train.py ... --no-transformer --exp-name radioml2018_trans_none

# 汇总
python report.py --runs-dir runs --out report.md
```

## 6. 4090 上的时间估算

按 4090 约 **3–6 ms/step**（batch 128–256，含前后向）估算，取中值 4 ms：

| 训练帧数 | 步/epoch | 1 epoch | 100 ep × 2 seeds | 说明 |
|---|---|---|---|---|
| 30,000（调试） | 234 | ~1 秒 | <1 分钟 | 仅验证流程 |
| 100,000 | 781 | ~3 秒 | 10 分钟 | 小规模消融够用 |
| 300,000 | 2,343 | ~9 秒 | 31 分钟 |  medium |
| 1,000,000 | 7,812 | ~31 秒 | 1.7 小时 | 接近论文规模 |
| 1,533,542（RadioML 全量 60%） | 11,980 | ~48 秒 | 2.7 小时 | 单组实验 |

**全部约 31 组实验 × 2 seeds ≈ 3–5 天连续运行**（早停通常在 60–100 epoch 触发，实际可能更短）。
其中 LSTM2 / MCLDNN 在 GPU 上收益最大（CPU 上分别是 TFRN 的 9 倍和 5.7 倍耗时，GPU 上差距会明显缩小）。

> 统计提示：2 seeds 下配对 t 检验（df=1）与 Wilcoxon（n=2，双尾最小 p=0.5）都无法给出 p<0.05。回复版本只报 mean±std，不要写显著性结论，也不要为此增加种子数或实验组数。

> 数据加载注意：STFT 是在 DataLoader worker 里现算的（0.13 ms/帧），8 workers 下约 25 秒/epoch（全量），与 GPU 计算时间同量级。如果 GPU 利用率长期低于 50%，把 `--num-workers` 提到 8 或 16。

## 7. 监控与排错

```bash
nvidia-smi -l 5                     # 看 GPU 利用率/显存
tail -f nohup.out                   # 或 tee 出来的 full_run.log
ls runs/*/results.json              # 完成的实验
python report.py --runs-dir runs    # 随时看已有结果的汇总表
```

常见问题：

- **OOM（系统内存）**：全量数据 21 GB，内存不足就用 `--max-samples 800000`，代码会按类别+SNR 均匀抽样并分块读取。
- **OOM（显存）**：把 `--batch-size` 从 256 降到 128。
- **GPU 利用率低**：增大 `--num-workers`；或把数据放本地 SSD（不要放网络盘）。
- **中途断了**：每个 seed 完成才写 `results.json`，重跑时可用 `--seeds <未完成的种子>` 续跑；checkpoint 已按 `runs/<exp>/seed<N>.pt` 保存。
- **想看某次评估**：`python evaluate.py --ckpt runs/<exp>/seed0.pt --data-path data/... --device cuda --plot per_snr.png`

## 8. 代码已做的云端适配

- 修复了全量数据被**重复加载两遍**、划分时再复制三份的峰值 ~42 GB 内存问题；现在只保留一份数组，Dataset 只存索引；
- `--max-samples` 改为按类别 + SNR 均匀抽样（RadioML 原始数据按调制/SNR 排序，直接截断会只剩一个类别）；
- 超过 8 GB 的 HDF5 采用分块读取，避免一次性分配。
