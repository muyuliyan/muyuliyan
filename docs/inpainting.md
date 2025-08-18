# 图像修复 (Inpainting) 详细配置指南

## 📋 概述

图像修复（Inpainting）任务旨在智能地填补图像中缺失、损坏或不需要的区域，广泛应用于照片修复、物体移除、图像编辑等领域。

## 🎯 支持的模型

### 1. EdgeConnect
- **特点**: 边缘引导的两阶段修复
- **适用场景**: 结构复杂的图像修复
- **优势**: 保持边缘完整性
- **配置示例**:
```yaml
model:
  name: "EdgeConnect"
  edge_model_path: "models/inpainting/edge_model.pth"
  inpaint_model_path: "models/inpainting/inpaint_model.pth"
  input_size: 256
  sigma: 2.0
```

### 2. PartialConv (Partial Convolution)
- **特点**: 基于部分卷积的单阶段修复
- **适用场景**: 不规则掩膜修复
- **优势**: 处理任意形状的洞
- **配置示例**:
```yaml
model:
  name: "PartialConv"
  freeze_enc_bn: false
  layer_size: 7
  checkpoint_path: "models/inpainting/partialconv.pth"
```

### 3. GMCNN (Gated Convolution)
- **特点**: 门控卷积机制
- **适用场景**: 自由形式的图像修复
- **优势**: 自动学习有效区域
- **配置示例**:
```yaml
model:
  name: "GMCNN"
  cnum: 32
  use_cuda: true
  use_lrelu: true
  checkpoint_path: "models/inpainting/gmcnn.pth"
```

### 4. LaMa (Large Mask Inpainting)
- **特点**: 基于Fast Fourier Convolution
- **适用场景**: 大面积掩膜修复
- **优势**: 处理大型缺失区域
- **配置示例**:
```yaml
model:
  name: "LaMa"
  architecture: "lama_fourier"
  ffc_kernel_size: 3
  ffc_activation: "lrelu"
  checkpoint_path: "models/inpainting/lama_model.pth"
```

### 5. MAT (Mask-Aware Transformer)
- **特点**: 基于Transformer的修复
- **适用场景**: 高分辨率图像修复
- **优势**: 全局感受野，细节丰富
- **配置示例**:
```yaml
model:
  name: "MAT"
  embed_dim: 512
  depth: 8
  num_heads: 8
  mlp_ratio: 4
  checkpoint_path: "models/inpainting/mat_model.pth"
```

## ⚙️ 完整配置文件

### 基础配置 (`config/inpainting_config.yaml`)

```yaml
# 任务基本信息
task: "inpainting"
version: "1.0"
description: "Image inpainting configuration"

# 模型配置
model:
  name: "EdgeConnect"
  input_size: 256
  checkpoint_path: "models/inpainting/edge_connect.pth"
  
  # 模型特定参数
  parameters:
    edge_threshold: 0.8
    sigma: 2.0
    use_spectral_norm: true
    
# 输入配置
input:
  image_path: "data/input/"
  mask_path: "data/input/masks/"
  supported_formats: [".jpg", ".jpeg", ".png", ".bmp"]
  
  # 掩膜处理配置
  mask_processing:
    auto_generate: false      # 自动生成掩膜
    mask_threshold: 127       # 掩膜二值化阈值
    dilate_kernel: 5          # 膨胀操作核大小
    erosion_iterations: 1     # 腐蚀操作次数
    smooth_mask: true         # 平滑掩膜边界
    
  # 预处理参数
  preprocessing:
    resize_input: true
    target_size: [256, 256]
    normalize: true
    mean: [0.5, 0.5, 0.5]
    std: [0.5, 0.5, 0.5]
    
# 输出配置
output:
  save_path: "data/output/inpainting/"
  format: "png"
  quality: 95
  
  # 后处理参数
  postprocessing:
    blend_boundary: true      # 边界融合
    blend_width: 10           # 融合宽度
    color_correction: true    # 颜色校正
    enhance_details: false    # 细节增强
    
# 处理配置
processing:
  device: "cuda"
  batch_size: 1
  num_workers: 4
  use_fp16: true
  
  # 多尺度处理
  multi_scale:
    enable: false
    scales: [256, 512]
    fusion_method: "pyramid"
    
  # 迭代修复
  iterative_refinement:
    enable: false
    num_iterations: 3
    refinement_model: "same"  # same, dedicated
    
# 掩膜生成配置（自动生成模式）
mask_generation:
  enable: false
  method: "random"  # random, object_removal, scratch_removal
  
  # 随机掩膜参数
  random_mask:
    num_holes: [1, 5]
    hole_size: [20, 100]
    brush_width: [10, 25]
    
  # 物体移除掩膜
  object_removal:
    detection_model: "yolo"
    target_classes: ["person", "car"]
    expansion_ratio: 1.2
    
# 验证配置
validation:
  enable: true
  metrics: ["PSNR", "SSIM", "LPIPS", "FID"]
  ground_truth_path: "data/validation/complete/"
  
# 日志配置
logging:
  level: "INFO"
  save_path: "logs/inpainting.log"
  tensorboard: true
  tensorboard_dir: "runs/inpainting"
  save_intermediate: false  # 保存中间结果
```

## 🚀 使用方法

### 1. 基础图像修复

```bash
python src/inpainting.py \
    --config config/inpainting_config.yaml \
    --input data/input/damaged_image.jpg \
    --mask data/input/masks/mask.png \
    --output data/output/restored_image.png
```

### 2. 批量修复处理

```bash
python src/inpainting.py \
    --config config/inpainting_config.yaml \
    --input_dir data/input/damaged_images/ \
    --mask_dir data/input/masks/ \
    --output_dir data/output/inpainting/ \
    --recursive true
```

### 3. 交互式物体移除

```bash
python src/interactive_inpainting.py \
    --config config/inpainting_config.yaml \
    --input data/input/image_with_object.jpg \
    --interactive true
```

### 4. 视频修复

```bash
python src/video_inpainting.py \
    --config config/inpainting_config.yaml \
    --input data/input/damaged_video.mp4 \
    --mask data/input/video_masks/ \
    --output data/output/restored_video.mp4 \
    --temporal_consistency true
```

## 🎭 不同应用场景的配置

### 1. 物体移除

```yaml
object_removal:
  model:
    name: "LaMa"
    
  input:
    mask_processing:
      auto_generate: true
      dilate_kernel: 15  # 更大的膨胀核
      
  output:
    postprocessing:
      blend_boundary: true
      blend_width: 20
      
  processing:
    iterative_refinement:
      enable: true
      num_iterations: 2
```

### 2. 老照片修复

```yaml
old_photo_restoration:
  model:
    name: "EdgeConnect"
    
  input:
    preprocessing:
      enhance_contrast: true
      gamma_correction: true
      gamma: 1.2
      
  output:
    postprocessing:
      vintage_color_correction: true
      noise_reduction: true
      
  mask_generation:
    method: "scratch_removal"
    auto_detect_scratches: true
```

### 3. 艺术品修复

```yaml
artwork_restoration:
  model:
    name: "MAT"
    
  processing:
    use_fp16: false  # 高精度处理
    multi_scale:
      enable: true
      scales: [512, 1024]
      
  validation:
    metrics: ["PSNR", "SSIM", "VGG_Loss"]
    expert_evaluation: true
```

### 4. 医学图像修复

```yaml
medical_inpainting:
  model:
    name: "PartialConv"
    
  input:
    preprocessing:
      preserve_intensity_range: true
      normalization_method: "z_score"
      
  processing:
    conservative_inpainting: true
    uncertainty_estimation: true
    
  output:
    quality_assessment: true
    clinical_validation: true
```

## 🎛️ 高级配置选项

### 1. 多阶段修复

```yaml
multi_stage_inpainting:
  enable: true
  stages:
    - name: "structure_reconstruction"
      model: "EdgeConnect"
      focus: "edges"
    - name: "texture_synthesis"
      model: "GMCNN"
      focus: "texture"
    - name: "refinement"
      model: "MAT"
      focus: "details"
```

### 2. 自适应修复策略

```yaml
adaptive_inpainting:
  enable: true
  strategy_selection:
    small_holes: "PartialConv"    # < 10% image area
    medium_holes: "LaMa"          # 10-50% image area
    large_holes: "MAT"            # > 50% image area
    
  hole_size_threshold: [0.1, 0.5]  # 面积比例阈值
```

### 3. 上下文感知修复

```yaml
context_aware_inpainting:
  enable: true
  context_analysis:
    semantic_segmentation: true
    texture_analysis: true
    color_distribution: true
    
  region_based_processing:
    enable: true
    region_detection_model: "mask_rcnn"
    per_region_strategy: true
```

### 4. 时序一致性（视频修复）

```yaml
temporal_consistency:
  enable: true
  optical_flow:
    method: "flownet"
    model_path: "models/flow/flownet.pth"
    
  consistency_loss:
    weight: 0.1
    temporal_window: 5
    
  post_processing:
    temporal_smoothing: true
    flicker_reduction: true
```

## 📊 质量评估和指标

### 评估配置

```yaml
evaluation:
  metrics:
    # 像素级指标
    - name: "PSNR"
      higher_better: true
      mask_aware: true  # 仅在修复区域计算
      
    - name: "SSIM"
      higher_better: true
      mask_aware: true
      
    # 感知质量指标
    - name: "LPIPS"
      higher_better: false
      network: "alex"
      mask_aware: true
      
    - name: "FID"
      higher_better: false
      feature_extractor: "inception_v3"
      
    # 专用修复指标
    - name: "P-IQA"  # Perceptual Image Quality Assessment
      higher_better: true
      
    - name: "U-IQA"  # Unpaired Image Quality Assessment
      higher_better: true
      
  visualization:
    save_comparison: true
    show_mask_overlay: true
    highlight_inpainted_region: true
    generate_before_after: true
```

## 🔧 训练配置

### 训练参数

```yaml
training:
  enable: false
  
  # 数据集配置
  dataset:
    complete_images_path: "data/train/complete/"
    mask_dataset_path: "data/train/masks/"
    validation_complete: "data/val/complete/"
    validation_masks: "data/val/masks/"
    
    # 数据增强
    augmentation:
      enable: true
      geometric_transforms: true
      color_transforms: false  # 保持颜色一致性
      mask_augmentation: true   # 掩膜变形
      
  # 损失函数配置
  loss:
    reconstruction_loss:
      type: "L1"
      weight: 1.0
      mask_weighted: true
      
    perceptual_loss:
      type: "VGG19"
      layers: ["relu2_2", "relu3_4", "relu4_4", "relu5_4"]
      weights: [1.0, 1.0, 1.0, 1.0]
      weight: 0.1
      
    style_loss:
      enable: true
      weight: 120.0
      
    adversarial_loss:
      type: "hinge"
      weight: 0.01
      discriminator_steps: 1
      
    total_variation_loss:
      enable: true
      weight: 0.1
      
  # 训练策略
  training_strategy:
    progressive_growing: false
    curriculum_learning: true
    mask_size_curriculum: [0.1, 0.2, 0.3, 0.4, 0.5]
    
  # 优化器配置
  optimizer:
    generator:
      type: "Adam"
      lr: 1e-4
      beta1: 0.0
      beta2: 0.9
      
    discriminator:
      type: "Adam"
      lr: 4e-4
      beta1: 0.0
      beta2: 0.9
```

## 📋 实用工具和脚本

### 1. 掩膜生成工具 (`tools/mask_generator.py`)

```python
import cv2
import numpy as np
from scipy import ndimage

class MaskGenerator:
    @staticmethod
    def generate_random_mask(image_size, hole_range=(20, 100), num_holes=(1, 5)):
        """生成随机掩膜"""
        h, w = image_size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        num_holes = np.random.randint(num_holes[0], num_holes[1] + 1)
        
        for _ in range(num_holes):
            hole_size = np.random.randint(hole_range[0], hole_range[1])
            x = np.random.randint(0, w - hole_size)
            y = np.random.randint(0, h - hole_size)
            
            mask[y:y+hole_size, x:x+hole_size] = 255
            
        return mask
    
    @staticmethod
    def generate_brush_mask(image_size, brush_width=(10, 25), length_range=(100, 300)):
        """生成画笔风格掩膜"""
        h, w = image_size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        num_strokes = np.random.randint(1, 6)
        
        for _ in range(num_strokes):
            width = np.random.randint(brush_width[0], brush_width[1])
            length = np.random.randint(length_range[0], length_range[1])
            
            # 随机起点和方向
            start_x = np.random.randint(width, w - width)
            start_y = np.random.randint(width, h - width)
            angle = np.random.uniform(0, 2 * np.pi)
            
            # 绘制画笔轨迹
            for i in range(length):
                x = int(start_x + i * np.cos(angle))
                y = int(start_y + i * np.sin(angle))
                
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(mask, (x, y), width//2, 255, -1)
                    
        return mask
    
    @staticmethod
    def object_removal_mask(image, object_class="person"):
        """基于目标检测生成物体移除掩膜"""
        # 这里需要集成目标检测模型
        # 示例代码框架
        detector = load_object_detector()
        detections = detector.detect(image, classes=[object_class])
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            mask[y1:y2, x1:x2] = 255
            
        # 膨胀掩膜以确保完全覆盖
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        return mask
```

### 2. 修复质量评估工具 (`tools/inpainting_metrics.py`)

```python
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from lpips import LPIPS

class InpaintingMetrics:
    def __init__(self):
        self.lpips_net = LPIPS(net='alex')
        
    def calculate_mask_aware_psnr(self, gt_image, pred_image, mask):
        """计算掩膜区域的PSNR"""
        mask_region = mask > 127
        
        gt_masked = gt_image[mask_region]
        pred_masked = pred_image[mask_region]
        
        mse = np.mean((gt_masked - pred_masked) ** 2)
        if mse == 0:
            return float('inf')
            
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        return psnr
    
    def calculate_mask_aware_ssim(self, gt_image, pred_image, mask):
        """计算掩膜区域的SSIM"""
        mask_region = mask > 127
        
        # 为SSIM计算创建掩膜窗口
        gt_masked = gt_image.copy()
        pred_masked = pred_image.copy()
        
        gt_masked[~mask_region] = 0
        pred_masked[~mask_region] = 0
        
        ssim = structural_similarity(
            gt_masked, pred_masked, 
            multichannel=True, 
            data_range=255
        )
        
        return ssim
    
    def calculate_inpainting_quality_score(self, original, inpainted, mask):
        """计算综合修复质量分数"""
        # 边界一致性
        boundary_score = self._calculate_boundary_consistency(
            original, inpainted, mask
        )
        
        # 纹理连续性
        texture_score = self._calculate_texture_continuity(
            original, inpainted, mask
        )
        
        # 色彩和谐性
        color_score = self._calculate_color_harmony(
            original, inpainted, mask
        )
        
        # 综合分数
        quality_score = (
            0.4 * boundary_score + 
            0.4 * texture_score + 
            0.2 * color_score
        )
        
        return quality_score
    
    def _calculate_boundary_consistency(self, original, inpainted, mask):
        """计算边界一致性"""
        # 获取掩膜边界
        kernel = np.ones((3, 3), np.uint8)
        boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        
        # 在边界区域比较原图和修复图
        boundary_region = boundary > 0
        
        if not np.any(boundary_region):
            return 1.0
            
        orig_boundary = original[boundary_region]
        inpaint_boundary = inpainted[boundary_region]
        
        # 计算颜色差异
        color_diff = np.mean(np.abs(orig_boundary - inpaint_boundary))
        consistency_score = max(0, 1 - color_diff / 255.0)
        
        return consistency_score
```

### 3. 批量处理脚本 (`scripts/batch_inpainting.py`)

```python
import os
import yaml
import argparse
from tqdm import tqdm
from src.inpainting import Inpainting

def batch_inpainting(config_path, input_dir, mask_dir, output_dir):
    """批量图像修复处理"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    inpainter = Inpainting(config)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    results = []
    
    for img_file in tqdm(image_files, desc="Processing images"):
        # 构建文件路径
        img_path = os.path.join(input_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)
        output_path = os.path.join(output_dir, f"restored_{img_file}")
        
        # 检查掩膜文件是否存在
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {img_file}, skipping...")
            continue
        
        try:
            # 执行修复
            result = inpainter.process(img_path, mask_path)
            inpainter.save(result, output_path)
            
            results.append({
                'input': img_file,
                'output': f"restored_{img_file}",
                'status': 'success'
            })
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            results.append({
                'input': img_file,
                'output': None,
                'status': 'failed',
                'error': str(e)
            })
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--input_dir", required=True, help="输入图像目录")
    parser.add_argument("--mask_dir", required=True, help="掩膜目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    
    args = parser.parse_args()
    
    results = batch_inpainting(
        args.config, 
        args.input_dir, 
        args.mask_dir, 
        args.output_dir
    )
    
    # 统计结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    total_count = len(results)
    
    print(f"\n处理完成:")
    print(f"成功: {success_count}/{total_count}")
    print(f"失败: {total_count - success_count}/{total_count}")
```

## 🚨 常见问题和解决方案

### 1. 修复结果不自然
```yaml
# 解决方案：增强感知损失
training:
  loss:
    perceptual_loss:
      weight: 0.2  # 增加感知损失权重
    style_loss:
      weight: 150.0  # 增加风格损失权重
```

### 2. 边界不连续
```yaml
# 解决方案：改善边界处理
output:
  postprocessing:
    blend_boundary: true
    blend_width: 15  # 增加融合宽度
    feather_edge: true  # 羽化边缘
```

### 3. 颜色不匹配
```yaml
# 解决方案：颜色校正
output:
  postprocessing:
    color_correction: true
    match_histogram: true  # 直方图匹配
    preserve_chrominance: true
```

### 4. 大面积修复质量差
```yaml
# 解决方案：使用专门的大掩膜模型
model:
  name: "LaMa"  # 专门为大掩膜设计
  
processing:
  multi_scale:
    enable: true
    scales: [256, 512, 1024]  # 多尺度处理
```

---

**提示**: 图像修复是一个复杂的任务，不同的图像内容和掩膜类型可能需要不同的模型和参数。建议根据具体应用场景选择合适的配置。