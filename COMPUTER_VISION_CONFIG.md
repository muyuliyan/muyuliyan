# 计算机视觉任务配置指南

本仓库支持三种主要的计算机视觉任务：超分辨率（Super-Resolution）、图像去噪（Denoising）和图像修复（Inpainting）。

## 📋 目录

- [快速开始](#快速开始)
- [任务配置](#任务配置)
  - [超分辨率 (Super-Resolution)](#超分辨率-super-resolution)
  - [图像去噪 (Denoising)](#图像去噪-denoising)
  - [图像修复 (Inpainting)](#图像修复-inpainting)
- [环境设置](#环境设置)
- [模型下载](#模型下载)
- [使用示例](#使用示例)

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆仓库
git clone https://github.com/muyuliyan/muyuliyan.git
cd muyuliyan

# 创建虚拟环境
python -m venv cv_env
source cv_env/bin/activate  # Linux/Mac
# cv_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置文件结构

```
muyuliyan/
├── config/
│   ├── super_resolution_config.yaml
│   ├── denoising_config.yaml
│   └── inpainting_config.yaml
├── models/
│   ├── super_resolution/
│   ├── denoising/
│   └── inpainting/
├── data/
│   ├── input/
│   └── output/
└── src/
    ├── super_resolution.py
    ├── denoising.py
    └── inpainting.py
```

## ⚙️ 任务配置

### 超分辨率 (Super-Resolution)

**用途**: 将低分辨率图像提升为高分辨率图像

**配置文件**: `config/super_resolution_config.yaml`

```yaml
task: "super_resolution"
model:
  name: "ESRGAN"  # 可选: ESRGAN, SRGAN, EDSR, RDN
  scale_factor: 4  # 放大倍数: 2, 4, 8
  checkpoint_path: "models/super_resolution/esrgan_x4.pth"

input:
  image_path: "data/input/"
  supported_formats: [".jpg", ".jpeg", ".png", ".bmp"]

output:
  save_path: "data/output/super_resolution/"
  format: "png"
  quality: 95

processing:
  batch_size: 1
  device: "cuda"  # cuda/cpu
  use_fp16: true
```

**运行命令**:
```bash
python src/super_resolution.py --config config/super_resolution_config.yaml --input data/input/low_res.jpg
```

### 图像去噪 (Denoising)

**用途**: 去除图像中的噪声，提高图像质量

**配置文件**: `config/denoising_config.yaml`

```yaml
task: "denoising"
model:
  name: "DnCNN"  # 可选: DnCNN, FFDNet, CBDNet, MPRNet
  noise_level: 25  # 噪声等级: 15, 25, 50
  checkpoint_path: "models/denoising/dncnn_25.pth"

input:
  image_path: "data/input/"
  supported_formats: [".jpg", ".jpeg", ".png", ".bmp"]
  
output:
  save_path: "data/output/denoising/"
  format: "png"
  
processing:
  batch_size: 4
  device: "cuda"
  patch_size: 256  # 处理图像块大小
  overlap: 32      # 重叠区域大小
```

**运行命令**:
```bash
python src/denoising.py --config config/denoising_config.yaml --input data/input/noisy_image.jpg
```

### 图像修复 (Inpainting)

**用途**: 填补图像中缺失或损坏的区域

**配置文件**: `config/inpainting_config.yaml`

```yaml
task: "inpainting"
model:
  name: "EdgeConnect"  # 可选: EdgeConnect, PartialConv, GMCNN, LaMa
  checkpoint_path: "models/inpainting/edge_connect.pth"

input:
  image_path: "data/input/"
  mask_path: "data/input/masks/"  # 掩膜文件路径
  supported_formats: [".jpg", ".jpeg", ".png", ".bmp"]

output:
  save_path: "data/output/inpainting/"
  format: "png"

processing:
  batch_size: 1
  device: "cuda"
  mask_threshold: 127  # 掩膜二值化阈值
  dilate_kernel: 5     # 膨胀操作核大小
```

**运行命令**:
```bash
python src/inpainting.py --config config/inpainting_config.yaml --input data/input/damaged.jpg --mask data/input/masks/mask.png
```

## 🔧 环境设置

### 基础依赖

创建 `requirements.txt`:

```txt
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.21.0
Pillow>=8.3.0
PyYAML>=5.4.0
tqdm>=4.62.0
matplotlib>=3.4.0
scikit-image>=0.18.0
tensorboard>=2.7.0
```

### GPU 支持 (推荐)

```bash
# CUDA 11.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证GPU可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📥 模型下载

### 预训练模型下载脚本

创建 `download_models.py`:

```python
import os
import urllib.request
from tqdm import tqdm

def download_model(url, save_path):
    """下载预训练模型"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    urllib.request.urlretrieve(url, save_path)

# 超分辨率模型
download_model(
    "https://github.com/xinntao/ESRGAN/releases/download/v0.0.0/RRDB_ESRGAN_x4.pth",
    "models/super_resolution/esrgan_x4.pth"
)

# 去噪模型
download_model(
    "https://github.com/cszn/DnCNN/releases/download/v0.0.0/dncnn_25.pth",
    "models/denoising/dncnn_25.pth"
)

# 图像修复模型
download_model(
    "https://github.com/knazeri/edge-connect/releases/download/v1.0.0/edge_connect.pth",
    "models/inpainting/edge_connect.pth"
)
```

运行下载:
```bash
python download_models.py
```

## 💡 使用示例

### 批量处理示例

```bash
# 超分辨率批量处理
python src/super_resolution.py \
    --config config/super_resolution_config.yaml \
    --input_dir data/input/low_res_images/ \
    --output_dir data/output/super_resolution/

# 去噪批量处理
python src/denoising.py \
    --config config/denoising_config.yaml \
    --input_dir data/input/noisy_images/ \
    --output_dir data/output/denoising/

# 图像修复批量处理
python src/inpainting.py \
    --config config/inpainting_config.yaml \
    --input_dir data/input/damaged_images/ \
    --mask_dir data/input/masks/ \
    --output_dir data/output/inpainting/
```

### Python API 使用

```python
from src.super_resolution import SuperResolution
from src.denoising import Denoising
from src.inpainting import Inpainting

# 超分辨率
sr = SuperResolution(config_path="config/super_resolution_config.yaml")
hr_image = sr.process("data/input/low_res.jpg")
sr.save(hr_image, "data/output/high_res.png")

# 去噪
denoiser = Denoising(config_path="config/denoising_config.yaml")
clean_image = denoiser.process("data/input/noisy.jpg")
denoiser.save(clean_image, "data/output/clean.png")

# 图像修复
inpainter = Inpainting(config_path="config/inpainting_config.yaml")
restored_image = inpainter.process("data/input/damaged.jpg", "data/input/mask.png")
inpainter.save(restored_image, "data/output/restored.png")
```

## 📊 性能优化

### 1. 批处理设置
- 超分辨率: batch_size=1 (内存消耗大)
- 去噪: batch_size=4-8 (根据GPU内存调整)
- 图像修复: batch_size=1-2 (模型复杂)

### 2. 内存优化
```yaml
processing:
  use_fp16: true      # 使用半精度浮点数
  gradient_checkpointing: true  # 梯度检查点
  max_image_size: 2048  # 限制最大图像尺寸
```

### 3. 多GPU支持
```yaml
processing:
  multi_gpu: true
  gpu_ids: [0, 1, 2, 3]  # 指定GPU设备
```

## 🔍 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 解决方案: 减小batch_size或图像尺寸
   RuntimeError: CUDA out of memory
   ```

2. **模型文件未找到**
   ```bash
   # 解决方案: 检查模型路径或重新下载
   FileNotFoundError: No such file or directory: 'models/xxx.pth'
   ```

3. **依赖包版本冲突**
   ```bash
   # 解决方案: 创建新的虚拟环境
   pip install --upgrade package_name
   ```

## 📚 详细文档

- [超分辨率详细配置](docs/super_resolution.md)
- [图像去噪详细配置](docs/denoising.md)
- [图像修复详细配置](docs/inpainting.md)
- [环境设置详细说明](docs/setup.md)

## 📞 支持

如有问题请联系：
- 📧 Email: 1468256361@qq.com / liyangyan314@gmail.com
- 💬 CSDN: 2403_86007563

---

**注意**: 请确保有足够的GPU内存和存储空间来运行这些任务。建议使用NVIDIA GPU (8GB+ VRAM) 以获得最佳性能。