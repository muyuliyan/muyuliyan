# 环境设置和依赖配置指南

## 📋 概述

本文档详细说明了如何为超分辨率、去噪和图像修复任务设置完整的开发和运行环境。

## 🖥️ 系统要求

### 最小配置
- **操作系统**: Ubuntu 18.04+ / Windows 10+ / macOS 10.15+
- **Python**: 3.7+
- **内存**: 8GB RAM
- **存储**: 20GB 可用空间
- **GPU**: 可选，但强烈推荐

### 推荐配置
- **操作系统**: Ubuntu 20.04 LTS
- **Python**: 3.8-3.10
- **内存**: 16GB+ RAM
- **存储**: 50GB+ SSD
- **GPU**: NVIDIA RTX 3080+ / Tesla V100+ (8GB+ VRAM)

## 🐍 Python 环境设置

### 1. 使用 Conda（推荐）

```bash
# 安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建虚拟环境
conda create -n cv_tasks python=3.9
conda activate cv_tasks

# 安装基础包
conda install numpy scipy matplotlib pillow opencv
```

### 2. 使用 virtualenv

```bash
# 安装 virtualenv
pip install virtualenv

# 创建虚拟环境
python -m venv cv_env
source cv_env/bin/activate  # Linux/Mac
# cv_env\Scripts\activate    # Windows

# 升级 pip
pip install --upgrade pip
```

## 📦 依赖包安装

### 核心依赖文件 (`requirements.txt`)

```txt
# 深度学习框架
torch>=1.12.0
torchvision>=0.13.0
torchaudio>=0.12.0

# 图像处理
opencv-python>=4.6.0
Pillow>=9.0.0
scikit-image>=0.19.0
imageio>=2.19.0
albumentations>=1.2.0

# 数值计算
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0

# 配置和工具
PyYAML>=6.0
tqdm>=4.64.0
click>=8.0.0
colorama>=0.4.5

# 评估指标
lpips>=0.1.4
pytorch-fid>=0.3.0
pyiqa>=0.1.7

# 可视化和日志
tensorboard>=2.9.0
wandb>=0.12.0
seaborn>=0.11.0

# 实用工具
h5py>=3.7.0
lmdb>=1.3.0
requests>=2.28.0
```

### GPU 支持安装

#### CUDA 11.8 (推荐)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 11.7
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

#### ROCm (AMD GPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

#### MPS (Apple Silicon)
```bash
pip install torch torchvision torchaudio
```

### 验证GPU安装

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## 🏗️ 项目结构设置

### 创建项目目录结构

```bash
# 创建项目目录
mkdir -p muyuliyan_cv && cd muyuliyan_cv

# 创建子目录
mkdir -p {config,models,data,src,logs,scripts,tools,tests,docs}
mkdir -p data/{input,output,train,val,test}
mkdir -p data/output/{super_resolution,denoising,inpainting}
mkdir -p models/{super_resolution,denoising,inpainting}
mkdir -p logs/{super_resolution,denoising,inpainting}

# 创建初始文件
touch src/__init__.py
touch config/__init__.py
touch tools/__init__.py
```

### 完整项目结构

```
muyuliyan_cv/
├── config/                     # 配置文件
│   ├── super_resolution_config.yaml
│   ├── denoising_config.yaml
│   └── inpainting_config.yaml
├── models/                     # 预训练模型
│   ├── super_resolution/
│   ├── denoising/
│   └── inpainting/
├── data/                       # 数据目录
│   ├── input/                  # 输入数据
│   ├── output/                 # 输出结果
│   ├── train/                  # 训练数据
│   ├── val/                    # 验证数据
│   └── test/                   # 测试数据
├── src/                        # 源代码
│   ├── __init__.py
│   ├── super_resolution.py
│   ├── denoising.py
│   ├── inpainting.py
│   ├── models/                 # 模型定义
│   ├── utils/                  # 工具函数
│   └── metrics/                # 评估指标
├── scripts/                    # 脚本文件
│   ├── download_models.py
│   ├── train_models.py
│   └── evaluate_models.py
├── tools/                      # 工具程序
│   ├── data_preprocessing.py
│   ├── visualization.py
│   └── benchmarking.py
├── tests/                      # 测试文件
├── logs/                       # 日志文件
├── docs/                       # 文档
├── requirements.txt
├── setup.py
└── README.md
```

## ⚙️ 环境配置文件

### 1. 开发环境配置 (`environment.yml`)

```yaml
name: cv_tasks
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy=1.21
  - scipy=1.7
  - matplotlib=3.5
  - pillow=9.0
  - opencv=4.6
  - pytorch=1.12
  - torchvision=0.13
  - torchaudio=0.12
  - cudatoolkit=11.8  # 如果使用GPU
  - pip
  - pip:
    - lpips>=0.1.4
    - pytorch-fid>=0.3.0
    - albumentations>=1.2.0
    - wandb>=0.12.0
    - pyiqa>=0.1.7
```

使用 Conda 创建环境：
```bash
conda env create -f environment.yml
conda activate cv_tasks
```

### 2. Docker 配置 (`Dockerfile`)

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace

# 安装Python依赖
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 安装PyTorch with CUDA
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 复制项目文件
COPY . .

# 设置Python路径
ENV PYTHONPATH=/workspace

# 暴露端口（如果需要web服务）
EXPOSE 8080

# 默认命令
CMD ["bash"]
```

Docker 使用命令：
```bash
# 构建镜像
docker build -t cv_tasks:latest .

# 运行容器
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    -p 8080:8080 \
    cv_tasks:latest
```

### 3. VS Code 配置 (`.vscode/settings.json`)

```json
{
    "python.defaultInterpreterPath": "./cv_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "python.sortImports.args": ["--profile", "black"],
    "files.associations": {
        "*.yaml": "yaml",
        "*.yml": "yaml"
    },
    "yaml.schemas": {
        "./config/schema.json": ["config/*.yaml", "config/*.yml"]
    },
    "editor.rulers": [88],
    "editor.formatOnSave": true
}
```

## 🔧 开发工具配置

### 1. 代码格式化 (`.pre-commit-config.yaml`)

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        
  - repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
      - id: black
        language_version: python3
        
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
        args: ["--profile", "black"]
        
  - repo: https://github.com/pycqa/flake8
    rev: 5.0.4
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]
```

安装和使用：
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### 2. 测试配置 (`pytest.ini`)

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=src
    --cov-report=html
    --cov-report=term-missing
markers =
    slow: marks tests as slow
    gpu: marks tests as requiring GPU
    integration: marks tests as integration tests
```

### 3. 性能分析配置

```python
# profiling_config.py
import cProfile
import pstats
from functools import wraps

def profile_function(func):
    """性能分析装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # 显示前20个最耗时的函数
        
        return result
    return wrapper
```

## 📊 监控和日志配置

### 1. 日志配置 (`logging_config.yaml`)

```yaml
version: 1
formatters:
  default:
    format: '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    datefmt: '%Y-%m-%d %H:%M:%S'
  detailed:
    format: '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s'
    datefmt: '%Y-%m-%d %H:%M:%S'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout
    
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    
  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: logs/error.log
    maxBytes: 10485760
    backupCount: 5

loggers:
  '':
    level: DEBUG
    handlers: [console, file, error_file]
    propagate: no
    
  src:
    level: DEBUG
    handlers: [console, file]
    propagate: no
```

### 2. W&B (Weights & Biases) 配置

```python
# wandb_config.py
import wandb

def init_wandb(config, project_name="cv_tasks"):
    """初始化W&B日志"""
    wandb.init(
        project=project_name,
        config=config,
        name=f"{config['task']}_{config['model']['name']}",
        tags=[config['task'], config['model']['name']],
        notes=config.get('description', ''),
        save_code=True
    )
    
    # 监控GPU使用率
    wandb.config.update({
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    })

def log_metrics(metrics, step=None):
    """记录指标"""
    wandb.log(metrics, step=step)

def log_images(images, caption="Results", step=None):
    """记录图像"""
    wandb.log({"images": [wandb.Image(img, caption=caption) for img in images]}, step=step)
```

## 🚀 自动化脚本

### 1. 环境安装脚本 (`setup.sh`)

```bash
#!/bin/bash

set -e

echo "Setting up CV Tasks environment..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.7"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

# 创建虚拟环境
echo "Creating virtual environment..."
python3 -m venv cv_env
source cv_env/bin/activate

# 升级pip
echo "Upgrading pip..."
pip install --upgrade pip

# 安装依赖
echo "Installing dependencies..."
pip install -r requirements.txt

# 检查GPU支持
echo "Checking GPU support..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 创建必要目录
echo "Creating project directories..."
mkdir -p data/{input,output,train,val,test}
mkdir -p data/output/{super_resolution,denoising,inpainting}
mkdir -p models/{super_resolution,denoising,inpainting}
mkdir -p logs/{super_resolution,denoising,inpainting}

# 下载模型权重（可选）
echo "Do you want to download pre-trained models? (y/n)"
read -r download_models
if [ "$download_models" = "y" ]; then
    echo "Downloading models..."
    python3 scripts/download_models.py
fi

echo "Setup completed successfully!"
echo "To activate the environment, run: source cv_env/bin/activate"
```

### 2. 模型下载脚本 (`scripts/download_models.py`)

```python
import os
import sys
import requests
from tqdm import tqdm
import hashlib

def download_file(url, filename, expected_hash=None):
    """下载文件并验证hash"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 检查文件是否已存在
    if os.path.exists(filename):
        if expected_hash:
            with open(filename, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash == expected_hash:
                print(f"File {filename} already exists and is valid.")
                return
    
    print(f"Downloading {filename}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=os.path.basename(filename),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    # 验证文件hash
    if expected_hash:
        with open(filename, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash != expected_hash:
            print(f"Warning: Hash mismatch for {filename}")
            return False
    
    print(f"Successfully downloaded {filename}")
    return True

def main():
    """下载所有预训练模型"""
    models = {
        # 超分辨率模型
        "models/super_resolution/esrgan_x4.pth": {
            "url": "https://github.com/xinntao/ESRGAN/releases/download/v0.0.0/RRDB_ESRGAN_x4.pth",
            "hash": "aa0d6e2d7e6e6c7c5a5c5e5e5e5e5e5e"
        },
        
        # 去噪模型
        "models/denoising/dncnn_25.pth": {
            "url": "https://github.com/cszn/DnCNN/releases/download/v0.0.0/dncnn_25.pth",
            "hash": "bb0d6e2d7e6e6c7c5a5c5e5e5e5e5e5e"
        },
        
        # 图像修复模型
        "models/inpainting/lama_model.pth": {
            "url": "https://github.com/saic-mdal/lama/releases/download/v1.0.0/lama_model.pth",
            "hash": "cc0d6e2d7e6e6c7c5a5c5e5e5e5e5e5e"
        }
    }
    
    print("Downloading pre-trained models...")
    
    success_count = 0
    total_count = len(models)
    
    for filename, info in models.items():
        try:
            if download_file(info["url"], filename, info.get("hash")):
                success_count += 1
        except Exception as e:
            print(f"Failed to download {filename}: {str(e)}")
    
    print(f"\nDownload completed: {success_count}/{total_count} successful")
    
    if success_count < total_count:
        print("Some models failed to download. You can manually download them later.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

## 🔍 故障排除

### 常见问题和解决方案

#### 1. CUDA 版本不匹配
```bash
# 问题：CUDA版本与PyTorch不兼容
# 解决方案：重新安装匹配的PyTorch版本
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. 内存不足
```bash
# 问题：GPU内存不足
# 解决方案：减小batch size或使用CPU
export CUDA_VISIBLE_DEVICES=""  # 强制使用CPU
```

#### 3. 依赖包冲突
```bash
# 问题：包版本冲突
# 解决方案：创建新的虚拟环境
conda create -n cv_tasks_new python=3.9
conda activate cv_tasks_new
pip install -r requirements.txt
```

#### 4. 模型加载失败
```python
# 问题：模型权重不匹配
# 解决方案：使用strict=False加载
model.load_state_dict(torch.load(model_path), strict=False)
```

### 性能优化建议

1. **使用SSD存储**: 数据读写密集时使用SSD
2. **数据预处理**: 使用多进程预处理数据
3. **内存管理**: 定期清理GPU缓存
4. **批处理**: 合理设置batch size
5. **混合精度**: 使用FP16减少内存使用

### 环境变量配置

```bash
# ~/.bashrc 或 ~/.zshrc
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 设置OpenCV使用的线程数
export OMP_NUM_THREADS=4

# 设置PyTorch使用的线程数
export MKL_NUM_THREADS=4

# 禁用TensorFlow警告
export TF_CPP_MIN_LOG_LEVEL=2
```

---

**提示**: 环境配置是项目成功的基础，建议按照本指南逐步设置并测试每个环节。如遇问题，请检查系统要求和依赖版本是否匹配。