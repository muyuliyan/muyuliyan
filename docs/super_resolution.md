# 超分辨率 (Super-Resolution) 详细配置指南

## 📋 概述

超分辨率任务旨在将低分辨率图像转换为高分辨率图像，广泛应用于图像增强、视频处理、医学影像等领域。

## 🎯 支持的模型

### 1. ESRGAN (Enhanced Super-Resolution GAN)
- **特点**: 生成质量高，细节丰富
- **适用场景**: 自然图像、照片增强
- **放大倍数**: 4x
- **配置示例**:
```yaml
model:
  name: "ESRGAN"
  architecture: "RRDBNet"
  scale_factor: 4
  num_feat: 64
  num_block: 23
  num_grow_ch: 32
  checkpoint_path: "models/super_resolution/RRDB_ESRGAN_x4.pth"
```

### 2. Real-ESRGAN
- **特点**: 真实世界图像超分，处理退化更好
- **适用场景**: 老照片修复、监控视频增强
- **放大倍数**: 2x, 4x
- **配置示例**:
```yaml
model:
  name: "Real-ESRGAN"
  model_type: "RealESRGAN_x4plus"
  scale_factor: 4
  checkpoint_path: "models/super_resolution/RealESRGAN_x4plus.pth"
  use_face_enhance: true  # 面部增强
```

### 3. EDSR (Enhanced Deep Super-Resolution)
- **特点**: 速度快，资源消耗少
- **适用场景**: 实时处理、移动设备
- **放大倍数**: 2x, 3x, 4x
- **配置示例**:
```yaml
model:
  name: "EDSR"
  n_resblocks: 32
  n_feats: 256
  scale_factor: 4
  checkpoint_path: "models/super_resolution/edsr_x4.pth"
```

### 4. SwinIR
- **特点**: 基于Transformer，效果先进
- **适用场景**: 高质量图像处理
- **放大倍数**: 2x, 3x, 4x, 8x
- **配置示例**:
```yaml
model:
  name: "SwinIR"
  upscale: 4
  in_chans: 3
  img_size: 64
  window_size: 8
  img_range: 1.0
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 2
  checkpoint_path: "models/super_resolution/swinir_sr_x4.pth"
```

## ⚙️ 完整配置文件

### 基础配置 (`config/super_resolution_config.yaml`)

```yaml
# 任务基本信息
task: "super_resolution"
version: "1.0"
description: "Super-resolution configuration"

# 模型配置
model:
  name: "ESRGAN"
  scale_factor: 4
  checkpoint_path: "models/super_resolution/RRDB_ESRGAN_x4.pth"
  
  # 模型特定参数
  parameters:
    num_feat: 64
    num_block: 23
    num_grow_ch: 32
    
# 输入配置
input:
  image_path: "data/input/"
  supported_formats: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
  
  # 预处理参数
  preprocessing:
    normalize: true
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    
# 输出配置
output:
  save_path: "data/output/super_resolution/"
  format: "png"  # png, jpg, tiff
  quality: 95    # JPEG质量 (1-100)
  
  # 后处理参数
  postprocessing:
    clamp: true
    color_space: "RGB"  # RGB, BGR, YUV
    
# 处理配置
processing:
  device: "cuda"      # cuda, cpu, mps
  batch_size: 1       # 批处理大小
  num_workers: 4      # 数据加载线程数
  use_fp16: true      # 使用半精度
  
  # 内存优化
  memory_optimization:
    enable: true
    max_image_size: 2048  # 最大图像尺寸
    tile_size: 512        # 分块处理大小
    tile_overlap: 32      # 分块重叠大小
    
# 验证配置
validation:
  enable: true
  metrics: ["PSNR", "SSIM", "LPIPS"]
  reference_path: "data/validation/gt/"
  
# 日志配置
logging:
  level: "INFO"
  save_path: "logs/super_resolution.log"
  tensorboard: true
  tensorboard_dir: "runs/super_resolution"
```

## 🚀 使用方法

### 1. 单张图像处理

```bash
python src/super_resolution.py \
    --config config/super_resolution_config.yaml \
    --input data/input/low_res.jpg \
    --output data/output/high_res.png
```

### 2. 批量处理

```bash
python src/super_resolution.py \
    --config config/super_resolution_config.yaml \
    --input_dir data/input/low_res_images/ \
    --output_dir data/output/super_resolution/ \
    --recursive true
```

### 3. 视频处理

```bash
python src/video_super_resolution.py \
    --config config/super_resolution_config.yaml \
    --input data/input/video.mp4 \
    --output data/output/video_sr.mp4 \
    --fps 30
```

## 📊 性能优化策略

### 1. 内存优化

```yaml
processing:
  memory_optimization:
    enable: true
    max_image_size: 2048  # 限制输入图像最大尺寸
    tile_size: 512        # 分块处理，减少内存使用
    tile_overlap: 32      # 分块重叠，避免边界artifacts
    clear_cache: true     # 定期清理GPU缓存
```

### 2. 速度优化

```yaml
processing:
  use_fp16: true          # 半精度计算
  compile_model: true     # PyTorch 2.0编译优化
  channels_last: true     # 内存布局优化
  
model:
  use_onnx: true          # 使用ONNX推理引擎
  onnx_path: "models/super_resolution/esrgan_x4.onnx"
```

### 3. 多GPU并行

```yaml
processing:
  multi_gpu: true
  gpu_ids: [0, 1, 2, 3]
  parallel_strategy: "data_parallel"  # data_parallel, model_parallel
```

## 🎛️ 高级配置选项

### 1. 自适应缩放

```yaml
adaptive_scaling:
  enable: true
  min_scale: 2
  max_scale: 8
  auto_detect: true       # 自动检测最佳缩放比例
  quality_threshold: 0.8  # 质量阈值
```

### 2. 渐进式增强

```yaml
progressive_enhancement:
  enable: true
  stages: [2, 4]          # 先2x后4x
  intermediate_save: true # 保存中间结果
```

### 3. 领域自适应

```yaml
domain_adaptation:
  enable: true
  source_domain: "natural"  # natural, anime, face, text
  target_domain: "photo"
  adaptation_model: "models/adaptation/nat2photo.pth"
```

## 📈 质量评估

### 评估指标配置

```yaml
evaluation:
  metrics:
    - name: "PSNR"
      higher_better: true
    - name: "SSIM"
      higher_better: true
    - name: "LPIPS"
      higher_better: false
      network: "alex"  # alex, vgg
    - name: "FID"
      higher_better: false
    - name: "NIQE"
      higher_better: false
      
  ground_truth_path: "data/validation/gt/"
  save_comparison: true
  comparison_path: "data/evaluation/comparisons/"
```

## 🔧 模型训练配置

### 训练参数

```yaml
training:
  enable: false  # 设为true启用训练模式
  
  # 数据集配置
  dataset:
    train_hr_path: "data/train/HR/"
    train_lr_path: "data/train/LR/"
    val_hr_path: "data/val/HR/"
    val_lr_path: "data/val/LR/"
    
  # 训练超参数
  hyperparameters:
    learning_rate: 1e-4
    batch_size: 16
    epochs: 1000
    warmup_epochs: 10
    
  # 损失函数
  loss:
    pixel_loss:
      type: "L1"  # L1, L2, Charbonnier
      weight: 1.0
    perceptual_loss:
      type: "VGG"
      layers: ["relu2_2", "relu3_4", "relu4_4"]
      weight: 0.1
    adversarial_loss:
      weight: 0.01
      
  # 优化器
  optimizer:
    type: "Adam"
    beta1: 0.9
    beta2: 0.999
    weight_decay: 0
    
  # 学习率调度
  scheduler:
    type: "CosineAnnealingLR"
    T_max: 1000
    eta_min: 1e-7
```

## 📋 实用脚本

### 1. 批量评估脚本 (`scripts/evaluate_sr.py`)

```python
import os
import yaml
from src.super_resolution import SuperResolution
from src.metrics import calculate_metrics

def evaluate_model(config_path, test_dir, gt_dir):
    """评估超分辨率模型性能"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    sr_model = SuperResolution(config)
    results = []
    
    for img_name in os.listdir(test_dir):
        # 处理图像
        lr_path = os.path.join(test_dir, img_name)
        sr_image = sr_model.process(lr_path)
        
        # 计算指标
        gt_path = os.path.join(gt_dir, img_name)
        metrics = calculate_metrics(sr_image, gt_path)
        results.append(metrics)
    
    return results

if __name__ == "__main__":
    results = evaluate_model(
        "config/super_resolution_config.yaml",
        "data/test/LR/",
        "data/test/HR/"
    )
    print(f"Average PSNR: {sum(r['PSNR'] for r in results) / len(results)}")
```

### 2. 模型转换脚本 (`scripts/convert_model.py`)

```python
import torch
import onnx
from src.models import load_model

def convert_to_onnx(config_path, output_path):
    """将PyTorch模型转换为ONNX格式"""
    model = load_model(config_path)
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # 导出ONNX模型
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    print(f"Model converted to {output_path}")

if __name__ == "__main__":
    convert_to_onnx(
        "config/super_resolution_config.yaml",
        "models/super_resolution/esrgan_x4.onnx"
    )
```

## 🚨 常见问题和解决方案

### 1. 内存不足
```yaml
# 解决方案：启用分块处理
processing:
  memory_optimization:
    enable: true
    tile_size: 256  # 减小分块大小
```

### 2. 处理速度慢
```yaml
# 解决方案：使用优化配置
processing:
  use_fp16: true
  compile_model: true
  batch_size: 4  # 增加批处理大小
```

### 3. 结果质量不佳
```yaml
# 解决方案：调整后处理参数
output:
  postprocessing:
    sharpen: true
    sharpen_factor: 0.5
    denoise: true
    denoise_strength: 0.1
```

---

**提示**: 根据具体的硬件配置和图像类型，可能需要调整上述参数以获得最佳性能和质量。