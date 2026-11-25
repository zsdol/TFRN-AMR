# TFRN — 完整实验代码（论文复现）

本目录是根据 `TFRN_revised.tex`（*TFRN: A Multi-Domain Modulation Recognition Framework with Multi-Scale Convolution and Cross-Attention Fusion*）实现的完整 PyTorch 实验代码，涵盖论文方法、全部基线、全部消融实验、评价指标与显著性检验。

## 1. 与论文逐项对应

| 论文位置 | 实现位置 | 关键设定 |
|---|---|---|
| §3 预备知识 / 式(2) | `data/stft.py` | STFT：Hamming 窗 N=40、hop L=20、L0=1024 → 40×50 幅度谱；M=floor((L0−N)/L)+1=50 |
| §4.1 总体结构 | `models/tfrn.py` | 双分支 + 融合 + FC；IQ 输出 16×128 token，TF 输出 1×128 CLS token，d=128 |
| §4.2 TFRN-IQ / 表1 / 式(4-6) | `models/mscr.py` | 预卷积 Conv1D(32,3,1,1)；6 个 MSCR 块（32→64→128 通道，每块时间减半）；Conv1 k=3/s=1、Conv2 k=3/s=2、Conv3 k=5/s=1 + MaxPool(3,2,1)；短连接 MaxPool(3,2,1)，跨stage用 1×1 投影；1024→16 |
| §4.3 TFRN-TF / 表2 / 式(7-12) | `models/tf_branch.py`, `models/transformer.py` | 3×(Conv2D 3×3 + BN + ReLU + MaxPool2d 2×2)，通道 32/64/128；展平为 P=30 个 patch token，Linear→128，+CLS +可学习位置编码；2 层 pre-LN Transformer（h=8, d_ff=256, FFN=ReLU），取 CLS 作为 X_tf |
| §4.4 特征融合 / 图2 | `models/fusion.py` | CAD（相加）、CST（自注意力）、**CCT（交叉注意力，采用）**：Q=X_tf(1 token)，K=V=X_IQ(16 tokens)，h_f=8，d_f=16，F=LN(X_tf+Z) |
| §5.1 数据集 | `data/datasets.py` | RadioML2018.01a（24 类/26 个 SNR/−20~30dB step2）与 5GRadio（6 类/11 个 SNR/step5）；HDF5 X/Y/Z 约定 |
| §5.2 实验设置 | `config.py`, `train.py` | 分层 6:2:2（按类别+SNR，固定划分种子）；AdamW lr=1e−4；ReduceLROnPlateau(×0.1, patience=10)；最多 200 epoch + 早停；交叉熵；多随机种子 |
| §5.2 指标 | `utils/metrics.py` | per-SNR / 平均 / 峰值 / 低信噪比([-20,0]dB)平均准确率 |
| §5.2 统计 | `utils/stats.py`, `report.py` | 配对 t 检验 + Wilcoxon 符号秩检验（mean±std over seeds） |
| §5.4 基线 | `models/baselines.py` | ResNet、MCNet、LSTM2、MCLDNN、FEA-T、VT-MCNet |
| §5.5 消融 | `train.py` 命令行 + `scripts/run_all.sh` | 单域(iq/tf)、融合(cad/cst/cct)、MSCR 深度(2/4/6/8)、No MultiScale、Transformer 头数(4/8/16)、No Trans |

## 2. 安装

建议使用 Python 3.10–3.12（PyTorch 官方 wheel 支持最完整）：

```bash
cd TFRN
python -m pip install -r requirements.txt
python smoke_test.py     # 结构/前向/反向/小样本训练全流程自检
```

## 3. 数据准备

- **RadioML2018.01a**：从 DeepSig 下载 HDF5（`GOLD_XYZ_OSC.0001_16.hdf5`），数据含 `X`(N,1024,2)、`Y`(N,24) one-hot、`Z`(N,1) SNR。类别顺序见 `config.py::RADIOML2018_CLASSES`。
- **5GRadio**：随论文代码发布的 HDF5，假定同样的 X/Y/Z 键（若键名不同，`data/datasets.py::_read_hdf5` 已兼容 `iq/labels/snr` 等常见命名）。
- 也可用 CSV 索引（`--csv-path`，列：`path,label,snr`，`path` 指向 (2,1024) 的 .npy 文件）。
- 无真实数据时 `--dataset synthetic` 会生成带 AWGN 的 M-PSK/QAM 合成数据用于冒烟测试（**不能用于报告论文数值**）。

## 4. 运行

### 主实验（以 2 个随机种子训练 TFRN，默认值）
```bash
python train.py --dataset radioml2018 --data-path GOLD_XYZ_OSC.0001_16.hdf5 \
  --model tfrn --fusion cct --seeds 0 1 --out-dir runs
python train.py --dataset 5gradio --data-path 5GRadio.hdf5 \
  --model tfrn --fusion cct --seeds 0 1 --out-dir runs
```

### 基线（完全相同协议）
```bash
python train.py --dataset radioml2018 --data-path GOLD_XYZ_OSC.0001_16.hdf5 \
  --model lstm2 --exp-name radioml2018_lstm2 --seeds 0 1
# 其他：resnet / mcnet / mcldnn / feat / vtmcnet
```

### 消融实验
```bash
# 单域分支
python train.py --dataset radioml2018 --data-path ... --model tfrn --mode iq   # TFRN-IQ
python train.py --dataset radioml2018 --data-path ... --model tfrn --mode tf   # TFRN-TF
# 融合机制
python train.py --dataset radioml2018 --data-path ... --model tfrn --fusion cad
python train.py --dataset radioml2018 --data-path ... --model tfrn --fusion cst
# MSCR 深度 / 无多尺度
python train.py --dataset radioml2018 --data-path ... --model tfrn --num-mscr 2
python train.py --dataset radioml2018 --data-path ... --model tfrn --no-multiscale
# Transformer 头数 / 无 Transformer
python train.py --dataset radioml2018 --data-path ... --model tfrn --tf-heads 4
python train.py --dataset radioml2018 --data-path ... --model tfrn --no-transformer
```

### 一键跑全套
编辑 `scripts/run_all.sh` 顶部两个数据文件路径，然后 `bash scripts/run_all.sh`。

### 评估与报告
```bash
python evaluate.py --ckpt runs/<exp>/seed0.pt --data-path ... --plot per_snr.png
python report.py --runs-dir runs --out report.md    # 汇总表 + 配对 t 检验 / Wilcoxon
```

结果保存在 `runs/<exp>/results.json`（每个种子的 per-SNR 准确率和训练曲线，以及 mean±std 汇总）。

## 5. 论文中标注 \\todo{} 的参数——本实现的假设

论文 LaTeX 源码多处数值仍为 `\todo{}`（作者待填），本代码采用如下**显式假设**，均可通过命令行覆盖：

| 论文位置 | 论文原文 | 本实现默认值 |
|---|---|---|
| 摘要/结论 | 平均准确率 ± std | 由实际运行填入 `results.json` |
| §4.2 表1 脚注 | 1×1 投影“verify against code” | **已实现**跨 stage 的 1×1 投影 |
| §4.3 表2 脚注 | pooling / P “verify against code” | 3 次 MaxPool2d(2,2) → P=30（40×50 → 5×6） |
| §5.1 RadioML | 全量或子集待定 | 支持全量；`--max-samples` 可截取子集 |
| §5.1 5GRadio | 帧数/信道/硬件损伤待定 | 假定随代码发布的 X/Y/Z HDF5 |
| §5.2 | batch size 待定 | 128（`--batch-size`） |
| §5.2 | 早停 patience 待定 | 20（`--early-stop-patience`） |
| §5.2 | 随机种子数待定 | 2（0~1，`--seeds`，按你的要求调整） |
| §5.2 | GPU/PyTorch 版本待定 | 自动检测 CUDA/CPU；requirements 中 torch≥2.0 |
| §4.4 | h_f 待定 | 8（`--fusion` 内部使用 `--tf-heads` 之外的独立配置 `CFG.fusion_heads=8`） |

## 6. 注意事项 / 已知边界

1. **基线为“按论文描述重实现”**：ResNet/MCNet/LSTM2/MCLDNN/FEA-T/VT-MCNet 的结构依照引用论文与本文描述实现，输入/输出接口统一。要与论文表格中的具体数值逐位对齐，应使用各基线作者的官方代码；本实现对所有模型使用**完全相同的数据划分与训练协议**，保证公平对比与消融有效性。
2. **STFT 归一化**：论文仅写“幅度谱 |X(m,k)|”，未说明是否取对数/归一化。本实现默认逐图除以最大值（`--tfi-norm max`），可用 `--tfi-log`、`--tfi-norm none|zscore` 切换；这是复现时最可能影响数值的设定之一。
3. **TF 输入尺寸**：默认 40×50（N=40, L=20, L0=1024）。TFBranch 会在初始化时根据实际 CNN 输出自动推断 patch 数 P，因此改变 STFT 参数不会破坏模型。
4. **MSCR 消融长度**：`num_mscr=2/4/6/8` 时 IQ token 长度分别为 256/64/16/4；通过 1×1 投影统一到 d=128 以兼容融合模块（默认 6 块时为恒等映射，与论文一致）。
5. **确定性**：`set_seed` 固定初始化与数据顺序；划分种子独立于模型种子；开启 CUDA 时设置了 cudnn deterministic（可能略慢）。
6. **FLOPs 为估算值**：`--complexity` 通过 hook 统计 Conv/Linear/MHA 的 MACs×2，注意力部分为近似公式，仅用于量级对比。
7. **2 个种子的统计效力**：默认 2 seeds 下，配对 t 检验（df=1）与 Wilcoxon 符号秩检验（n=2，双尾最小 p=0.5）**均无法给出 p<0.05 的显著性结论**。因此回复版本只报告 mean±std 即可，**不要写"statistically significant / p<0.05"**；论文中的 `\todo{p<0.05}` 建议改为描述性表述。基线与指标均未增加。

## 7. 能跑起来吗？要跑多久？

**能跑。** 代码已在本机（Python 3.13 / CPU-only / 6 线程）完整验证：
- `python smoke_test.py` 全部通过 → 用合成数据跑通前向/反向/训练全流程；
- 加上真实 HDF5 数据集即可跑正式实验（需要 RadioML2018.01a 和 5GRadio 文件）；
- 无 GPU 也能跑，只是慢；有 NVIDIA GPU 时需安装 CUDA 版 torch（当前环境装的是 CPU 版）。

### 实测耗时（本机 CPU，batch=128，前向+反向+优化器）

| 模型 | 单步耗时 |
|---|---|
| TFRN | 1.12 s |
| ResNet | 1.44 s |
| MCNet | 2.52 s |
| LSTM2 | 10.07 s |
| MCLDNN | 6.44 s |
| FEA-T | 0.05 s |
| VT-MCNet | 2.52 s |

STFT 生成 0.13 ms/帧，不是瓶颈；瓶颈是卷积/LSTM 的 CPU 计算。

### 单组实验耗时估算（TFRN，100 epoch × 2 seeds）

| 训练帧数 | 步/epoch | CPU 单机 | GPU 估计* |
|---|---|---|---|
| 6,000（快速试跑，2 epoch） | 47 | ~2 分钟 | <1 分钟 |
| 20,000 | 156 | 9.7 小时 | 7 分钟 |
| 100,000 | 781 | 2.0 天 | 31 分钟 |
| 300,000 | 2,343 | 6.1 天 | 1.6 小时 |
| 1,000,000 | 7,812 | 20.3 天 | 5.2 小时 |
| 1,533,542（RadioML 全量 60%） | 11,980 | 31.1 天 | 8 小时 |

\* GPU 按消费级显卡（RTX 3060 级别）12 ms/step 估算，实际取决于型号和数据加载。

**完整实验矩阵 ≈ 31 组（主实验 14 + 消融 17）× 2 seeds**：
- GPU：约 1.5~2 周连续运行；
- 本机 CPU：不可行（数月）。

### 实用建议

1. **先快速试跑**：`bash scripts/quick_demo.sh <你的 hdf5 路径>`，6000 帧/2 epoch/单种子，几分钟验证流程。
2. **用 `--max-samples` 做小规模实验**（已修复为按类别+SNR 均匀抽样，不会只抽到单一类别）。论文级结果仍需全量数据 + GPU。
3. **减少 seeds**：调试时 `--seeds 0`；正式报告用默认 2 个。
4. **内存**：全量 RadioML2018.01a 的 X 数组约 21 GB RAM，内存不足务必用 `--max-samples`（代码会分块读取）。
5. **有 GPU 的话**：安装 CUDA 版 torch 后加 `--device cuda`，并适当调大 `--batch-size`；LSTM2/MCLDNN 在 CPU 上尤其慢，GPU 上会快很多。

### 云服务器（RTX 4090）部署

如果使用 RTX 4090 云服务器跑全套实验，见 **`DEPLOY_4090.md`**：包含机型选择（内存建议 ≥64 GB）、CUDA 环境配置、数据上传、tmux 挂后台分批跑、时间/费用估算（全套约 3–5 天）与常见排错。代码已针对云端做了大内存优化（数据只加载一份、Dataset 只存索引、HDF5 分块读取）。
