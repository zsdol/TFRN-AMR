# TFRN

## 安装

建议使用 Python 3.10–3.12（PyTorch 官方 wheel 支持最完整）：

```bash
cd TFRN
python -m pip install -r requirements.txt
python smoke_test.py     # 结构/前向/反向/小样本训练全流程自检
```

## 数据准备

- **RadioML2018.01a**：从 DeepSig 下载 HDF5（`GOLD_XYZ_OSC.0001_16.hdf5`），数据含 `X`(N,1024,2)、`Y`(N,24) one-hot、`Z`(N,1) SNR。类别顺序见 `config.py::RADIOML2018_CLASSES`。
- **5GRadio**：随论文代码发布的 HDF5，假定同样的 X/Y/Z 键（若键名不同，`data/datasets.py::_read_hdf5` 已兼容 `iq/labels/snr` 等常见命名）。
- 也可用 CSV 索引（`--csv-path`，列：`path,label,snr`，`path` 指向 (2,1024) 的 .npy 文件）。


## 运行

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

