# 图像去噪 (Denoising) 详细配置指南

## 📋 概述

图像去噪任务旨在从噪声图像中恢复清晰的图像，是计算机视觉中的基础任务，广泛应用于医学影像、天文图像、监控视频等领域。

## 🎯 支持的模型

### 1. DnCNN (Denoising Convolutional Neural Network)
- **特点**: 经典去噪模型，效果稳定
- **适用场景**: 高斯噪声去除
- **噪声类型**: 高斯白噪声
- **配置示例**:
```yaml
model:
  name: "DnCNN"
  depth: 17
  num_channels: 64
  kernel_size: 3
  noise_level: 25  # 15, 25, 50
  checkpoint_path: "models/denoising/dncnn_25.pth"
```

### 2. FFDNet (Fast and Flexible Denoising Network)
- **特点**: 可控噪声等级，速度快
- **适用场景**: 实时去噪，可变噪声强度
- **噪声类型**: 高斯噪声、真实噪声
- **配置示例**:
```yaml
model:
  name: "FFDNet"
  num_input_channels: 3
  num_feature_maps: 64
  num_layers: 15
  flexible_noise_level: true
  checkpoint_path: "models/denoising/ffdnet_color.pth"
```

### 3. CBDNet (Toward Convolutional Blind Denoising)
- **特点**: 盲去噪，无需预知噪声类型
- **适用场景**: 真实世界图像去噪
- **噪声类型**: 真实噪声（包含各种噪声）
- **配置示例**:
```yaml
model:
  name: "CBDNet"
  in_channels: 3
  out_channels: 3
  num_features: 64
  num_stages: 4
  checkpoint_path: "models/denoising/cbdnet_real.pth"
```

### 4. MPRNet (Multi-Stage Progressive Image Restoration)
- **特点**: 多阶段渐进式恢复
- **适用场景**: 高质量图像恢复
- **噪声类型**: 多种噪声类型
- **配置示例**:
```yaml
model:
  name: "MPRNet"
  n_feat: 96
  scale_unetfeats: 48
  scale_orsnetfeats: 32
  num_cab: 8
  kernel_size: 3
  reduction: 4
  bias: false
  checkpoint_path: "models/denoising/mprnet_denoising.pth"
```

### 5. Restormer
- **特点**: 基于Transformer的去噪模型
- **适用场景**: 最新技术，效果优异
- **噪声类型**: 通用噪声
- **配置示例**:
```yaml
model:
  name: "Restormer"
  inp_channels: 3
  out_channels: 3
  dim: 48
  num_blocks: [4, 6, 6, 8]
  num_heads: [1, 2, 4, 8]
  ffn_expansion_factor: 2.66
  bias: false
  LayerNorm_type: "WithBias"
  checkpoint_path: "models/denoising/restormer_denoising.pth"
```

## ⚙️ 完整配置文件

### 基础配置 (`config/denoising_config.yaml`)

```yaml
# 任务基本信息
task: "denoising"
version: "1.0"
description: "Image denoising configuration"

# 模型配置
model:
  name: "DnCNN"
  noise_level: 25
  checkpoint_path: "models/denoising/dncnn_25.pth"
  
  # 模型特定参数
  parameters:
    depth: 17
    num_channels: 64
    kernel_size: 3
    use_bnorm: true
    
# 输入配置
input:
  image_path: "data/input/"
  supported_formats: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
  
  # 噪声配置（用于合成噪声数据）
  noise_synthesis:
    enable: false
    noise_type: "gaussian"  # gaussian, poisson, salt_pepper, speckle
    noise_parameters:
      sigma: 25
      intensity: 0.1
      
  # 预处理参数
  preprocessing:
    normalize: true
    to_ycbcr: false  # 转换到YCbCr色彩空间
    process_y_only: false  # 仅处理Y通道
    
# 输出配置
output:
  save_path: "data/output/denoising/"
  format: "png"
  quality: 95
  
  # 后处理参数
  postprocessing:
    clamp: true
    color_space_convert: false
    enhance_contrast: false
    
# 处理配置
processing:
  device: "cuda"
  batch_size: 4
  num_workers: 4
  use_fp16: true
  
  # 分块处理配置
  patch_processing:
    enable: true
    patch_size: 256
    overlap: 32
    padding_mode: "reflect"  # reflect, replicate, constant
    
  # 内存优化
  memory_optimization:
    enable: true
    max_image_size: 2048
    clear_cache_freq: 10
    
# 噪声估计配置
noise_estimation:
  enable: true
  method: "pca"  # pca, median, robust
  block_size: 8
  save_noise_map: false
  
# 验证配置
validation:
  enable: true
  metrics: ["PSNR", "SSIM", "LPIPS"]
  noise_free_reference: "data/validation/clean/"
  
# 日志配置
logging:
  level: "INFO"
  save_path: "logs/denoising.log"
  tensorboard: true
  tensorboard_dir: "runs/denoising"
```

## 🚀 使用方法

### 1. 单张图像去噪

```bash
python src/denoising.py \
    --config config/denoising_config.yaml \
    --input data/input/noisy_image.jpg \
    --output data/output/clean_image.png
```

### 2. 批量去噪处理

```bash
python src/denoising.py \
    --config config/denoising_config.yaml \
    --input_dir data/input/noisy_images/ \
    --output_dir data/output/denoising/ \
    --recursive true
```

### 3. 自适应噪声等级去噪

```bash
python src/denoising.py \
    --config config/denoising_config.yaml \
    --input data/input/noisy_image.jpg \
    --output data/output/clean_image.png \
    --auto_noise_estimation true
```

### 4. 视频去噪

```bash
python src/video_denoising.py \
    --config config/denoising_config.yaml \
    --input data/input/noisy_video.mp4 \
    --output data/output/clean_video.mp4 \
    --temporal_consistency true
```

## 📊 不同噪声类型的配置

### 1. 高斯噪声

```yaml
model:
  name: "DnCNN"
  noise_level: 25
  
input:
  noise_synthesis:
    noise_type: "gaussian"
    noise_parameters:
      sigma: 25
```

### 2. 真实世界噪声

```yaml
model:
  name: "CBDNet"
  
input:
  preprocessing:
    normalize: false  # 保持原始像素值范围
    
processing:
  patch_processing:
    patch_size: 512  # 更大的patch用于捕获噪声模式
```

### 3. 低光照噪声

```yaml
model:
  name: "Restormer"
  
input:
  preprocessing:
    gamma_correction: true
    gamma: 2.2
    
output:
  postprocessing:
    enhance_contrast: true
    contrast_factor: 1.2
```

### 4. 医学图像噪声

```yaml
model:
  name: "MPRNet"
  
input:
  preprocessing:
    to_ycbcr: false
    bit_depth: 16  # 支持16位图像
    
processing:
  use_fp16: false  # 医学图像需要高精度
```

## 🎛️ 高级配置选项

### 1. 多尺度处理

```yaml
multi_scale_processing:
  enable: true
  scales: [1.0, 0.8, 0.6]
  fusion_method: "weighted_average"  # weighted_average, selective
  weights: [0.5, 0.3, 0.2]
```

### 2. 时域一致性（视频去噪）

```yaml
temporal_consistency:
  enable: true
  frame_buffer_size: 5
  optical_flow: true
  flow_method: "farneback"  # farneback, lucas_kanade
  warp_method: "bilinear"
```

### 3. 渐进式去噪

```yaml
progressive_denoising:
  enable: true
  stages: 3
  noise_reduction_factors: [0.7, 0.5, 0.3]
  intermediate_save: false
```

### 4. 自适应处理

```yaml
adaptive_processing:
  enable: true
  noise_detection: true
  region_based: true
  region_size: 64
  threshold_method: "otsu"  # otsu, adaptive_mean, adaptive_gaussian
```

## 📈 质量评估和比较

### 评估指标配置

```yaml
evaluation:
  metrics:
    - name: "PSNR"
      higher_better: true
      range: [0, 50]
    - name: "SSIM"
      higher_better: true
      range: [0, 1]
    - name: "LPIPS"
      higher_better: false
      network: "alex"
    - name: "BRISQUE"  # 无参考质量评估
      higher_better: false
    - name: "NIQE"     # 无参考质量评估
      higher_better: false
      
  visualization:
    save_comparison: true
    comparison_layout: "grid"  # grid, side_by_side
    include_difference_map: true
    difference_amplification: 5
```

## 🔧 训练配置

### 训练参数

```yaml
training:
  enable: false
  
  # 数据集配置
  dataset:
    clean_images_path: "data/train/clean/"
    noisy_images_path: "data/train/noisy/"
    validation_clean: "data/val/clean/"
    validation_noisy: "data/val/noisy/"
    
    # 数据增强
    augmentation:
      enable: true
      horizontal_flip: true
      vertical_flip: true
      rotation: true
      color_jitter: false
      
  # 训练超参数
  hyperparameters:
    learning_rate: 1e-3
    batch_size: 32
    epochs: 500
    patience: 50  # 早停patience
    
  # 损失函数
  loss:
    type: "CharbonnierLoss"  # L1, L2, CharbonnierLoss, SSIM
    parameters:
      epsilon: 1e-3
      
  # 优化器
  optimizer:
    type: "Adam"
    beta1: 0.9
    beta2: 0.999
    weight_decay: 1e-8
    
  # 学习率调度
  scheduler:
    type: "ReduceLROnPlateau"
    factor: 0.5
    patience: 10
    min_lr: 1e-7
```

## 📋 实用工具和脚本

### 1. 噪声合成工具 (`tools/noise_synthesis.py`)

```python
import numpy as np
import cv2
from scipy import ndimage

class NoiseSynthesis:
    @staticmethod
    def add_gaussian_noise(image, sigma):
        """添加高斯噪声"""
        noise = np.random.normal(0, sigma, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_poisson_noise(image, intensity=1.0):
        """添加泊松噪声"""
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_speckle_noise(image, intensity=0.1):
        """添加斑点噪声"""
        gauss = np.random.randn(*image.shape) * intensity
        noisy = image + image * gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
```

### 2. 噪声估计工具 (`tools/noise_estimation.py`)

```python
import numpy as np
from sklearn.decomposition import PCA

class NoiseEstimation:
    @staticmethod
    def estimate_noise_pca(image, block_size=8):
        """使用PCA方法估计噪声"""
        h, w = image.shape[:2]
        blocks = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = image[i:i+block_size, j:j+block_size]
                blocks.append(block.flatten())
        
        blocks = np.array(blocks)
        pca = PCA(n_components=1)
        pca.fit(blocks)
        
        # 估计噪声标准差
        residuals = blocks - pca.inverse_transform(pca.transform(blocks))
        noise_std = np.std(residuals)
        
        return noise_std
    
    @staticmethod
    def estimate_noise_median(image, kernel_size=3):
        """使用中值滤波方法估计噪声"""
        filtered = ndimage.median_filter(image, size=kernel_size)
        noise_map = image - filtered
        noise_std = np.std(noise_map)
        return noise_std
```

### 3. 批量评估脚本 (`scripts/evaluate_denoising.py`)

```python
import os
import yaml
import cv2
import numpy as np
from src.denoising import Denoising
from src.metrics import calculate_psnr, calculate_ssim

def evaluate_denoising_model(config_path, test_noisy_dir, test_clean_dir):
    """评估去噪模型性能"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    denoiser = Denoising(config)
    results = []
    
    for img_name in os.listdir(test_noisy_dir):
        # 去噪处理
        noisy_path = os.path.join(test_noisy_dir, img_name)
        clean_pred = denoiser.process(noisy_path)
        
        # 读取真实清晰图像
        clean_path = os.path.join(test_clean_dir, img_name)
        clean_gt = cv2.imread(clean_path)
        
        # 计算指标
        psnr = calculate_psnr(clean_pred, clean_gt)
        ssim = calculate_ssim(clean_pred, clean_gt)
        
        results.append({
            'image': img_name,
            'PSNR': psnr,
            'SSIM': ssim
        })
    
    return results

if __name__ == "__main__":
    results = evaluate_denoising_model(
        "config/denoising_config.yaml",
        "data/test/noisy/",
        "data/test/clean/"
    )
    
    avg_psnr = np.mean([r['PSNR'] for r in results])
    avg_ssim = np.mean([r['SSIM'] for r in results])
    
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
```

## 🚨 常见问题和解决方案

### 1. 过度去噪问题
```yaml
# 解决方案：降低去噪强度
model:
  noise_level: 15  # 降低噪声等级

output:
  postprocessing:
    detail_enhancement: true
    enhancement_factor: 1.2
```

### 2. 细节丢失
```yaml
# 解决方案：使用多尺度处理
multi_scale_processing:
  enable: true
  detail_preservation: true
  edge_threshold: 0.1
```

### 3. 颜色偏移
```yaml
# 解决方案：单独处理亮度通道
input:
  preprocessing:
    to_ycbcr: true
    process_y_only: true  # 仅去噪Y通道，保持色度
```

### 4. 处理速度慢
```yaml
# 解决方案：优化处理参数
processing:
  patch_processing:
    patch_size: 128  # 减小patch大小
    overlap: 16      # 减小重叠区域
  
  use_fp16: true     # 使用半精度
  batch_size: 8      # 增加批处理大小
```

## 🎯 应用场景专用配置

### 1. 夜景摄影去噪
```yaml
night_photography:
  model:
    name: "Restormer"
  input:
    preprocessing:
      gamma_correction: true
      gamma: 1.8
  output:
    postprocessing:
      enhance_contrast: true
      preserve_highlights: true
```

### 2. 医学图像去噪
```yaml
medical_imaging:
  model:
    name: "MPRNet"
  processing:
    use_fp16: false  # 高精度处理
    preserve_intensity: true
  validation:
    metrics: ["PSNR", "CNR", "SNR"]  # 医学专用指标
```

### 3. 老照片修复
```yaml
old_photo_restoration:
  model:
    name: "CBDNet"
  input:
    preprocessing:
      histogram_equalization: true
  output:
    postprocessing:
      color_restoration: true
      vintage_enhancement: false
```

---

**提示**: 不同类型的噪声和应用场景可能需要不同的模型和参数配置。建议根据具体需求选择合适的配置。