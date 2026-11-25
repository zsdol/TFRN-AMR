# RTX 4090 云服务器
## 环境配置（服务器端）

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


