# Framework Summary

## 📋 Overview

This is a comprehensive multimodal fusion framework for audio-visual classification with support for multiple encoders, fusion methods, and training strategies including knowledge distillation.

## 🎯 Key Features

### ✅ Data Processing
- **Video Preprocessing**: 3 sampling methods (uniform, random, consecutive)
- **Data Splitting**: Stratified and random splits
- **Soft Label Generation**: Offline KD support for faster training

### ✅ Model Architectures

**Video Encoders (5 types)**:
- S3D (Spatiotemporal 3D CNN)
- X3D (Efficient 3D CNN with variants: XS, S, M, L)
- MoViNet (Mobile Video Networks, A0-A5)
- I3D (Inflated 3D ConvNet)
- Video MAE (Masked Autoencoder)

**Audio Encoders (6 types)**:
- ResNet-18
- ResNet-50
- MobileNet V2
- EfficientNet (B0-B7)
- PANN CNN10 (pretrained on AudioSet)
- PANN CNN14 (pretrained on AudioSet)

**Fusion Methods (5 types)**:
- Concat (Simple concatenation)
- Cross-Attention (Bi-directional)
- Gated Fusion (Learnable gates)
- Joint Cross-Attention (Multi-layer self + cross)
- MBT (Multimodal Bottleneck Transformer)

### ✅ Training Strategies
- **Standard Fusion Training**: Train full multimodal models
- **Single Modality**: Train video-only or audio-only
- **Knowledge Distillation**:
  - Online KD (teacher forward during training)
  - Offline KD (pre-computed soft labels, 2-3x faster)

### ✅ Logging & Metrics
- **WandB Integration**: Online experiment tracking (free tier)
- **Complexity Metrics**: Parameters, GFLOPs, model size
- **Evaluation Metrics**: Accuracy, precision, recall, F1, confusion matrix
- **Per-class Metrics**: Detailed class-wise performance

## 📁 Directory Structure

```
reorganized/
├── README.md                      # Main documentation
├── SETUP.md                       # Setup instructions
├── USAGE.md                       # Usage guide
├── FRAMEWORK_SUMMARY.md          # This file
├── requirements.txt              # Pip dependencies
├── environment.yml               # Conda environment
│
├── data/                         # Data processing
│   ├── preprocessing/            # Video/audio preprocessing
│   │   ├── video_preprocessing.py
│   │   └── soft_label_generator.py
│   ├── splits/                   # Data splitting
│   │   └── split_utils.py
│   └── datasets/                 # PyTorch datasets
│       └── multimodal_dataset.py
│
├── models/                       # Model architectures
│   ├── encoders/                 # Video and audio encoders
│   │   ├── video_encoders.py     # 5 video encoder types
│   │   └── audio_encoders.py     # 6 audio encoder types
│   └── fusion/                   # Fusion methods
│       └── fusion_methods.py     # 5 fusion strategies
│
├── training/                     # Training modules
│   ├── fusion_training/          # Multimodal fusion
│   ├── single_training/          # Single modality
│   └── knowledge_distillation/   # KD training
│
├── utils/                        # Utilities
│   ├── metrics/                  # Evaluation metrics
│   │   └── metrics.py
│   ├── logging/                  # Logging (WandB)
│   │   └── wandb_logger.py
│   └── complexity/               # Parameter & GFLOPs
│       └── complexity_analysis.py
│
├── configs/                      # Configuration files
│   ├── example_joint_crossattn.yaml
│   └── example_kd_student.yaml
│
├── scripts/                      # Executable scripts
│   ├── train_fusion.py          # Main training script
│   └── generate_report.py       # Report generation
│
├── docs/                         # Additional documentation
├── pretrained/                   # (Optional) Local weights directory
├── checkpoints/                  # Training checkpoints
└── logs/                         # Training logs
```

## 🚀 Quick Start (3 Steps)

### 1. Setup Environment

```bash
cd /home/hoang.pm/reorganized
conda env create -f environment.yml
conda activate fusion_framework
```

### 2. Prepare Data

```bash
# Preprocess videos
python data/preprocessing/video_preprocessing.py \
    --input_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/video_dataset \
    --output_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video \
    --sampling_method uniform \
    --num_frames 16

# Preprocess audio
python data/preprocessing/audio_preprocessing.py \
    --input_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/audio_dataset \
    --output_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio \
    --sample_rate 32000 \
    --duration 2.0

# Create fixed splits
python data/splits/split_utils.py --create \
    --video_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video \
    --audio_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio \
    --output_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/splits \
    --seed 42 \
    --test_per_class 700 \
    --val_per_class 700
```

### 2.1 Download Pretrained Models

```bash
python download_pretrained_models.py --all
```

Weights are stored in:
`/mnt/disk1/backup_user/hoang.pm/pretrained_models/`

### 3. Train Model

```bash
# Train fusion model
python scripts/train_fusion.py \
    --config configs/example_joint_crossattn.yaml
```

### 4. Run All Video-Only Encoders

```bash
python scripts/run_all_video_experiments.py \
    --video_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video_uniform \
    --audio_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio \
    --split_file /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/splits_uniform/splits.json \
    --wandb_project "Multimodalities for Aquaculture" \
    --wandb_entity "hoangpmh2406-vinuniversity" \
    --epochs 50
```

## 📊 Model Comparison

| Configuration | Video | Audio | Fusion | Params | Use Case |
|--------------|-------|-------|--------|--------|----------|
| **High Performance** | S3D | PANN CNN14 | Joint Cross-Attn | ~52M | Best accuracy |
| **Balanced** | S3D | PANN CNN10 | Cross-Attn | ~45M | Good accuracy/speed |
| **Efficient** | MoViNet-A0 | MobileNet V2 | Concat | ~12M | Mobile/edge |
| **Ultra Efficient (KD)** | MoViNet-A0 | MobileNet V2 | Cross-Attn | ~9M | Edge devices |

## 🎓 Knowledge Distillation Workflow

### Why Use KD?
- **2-3x faster training** with offline mode
- **Smaller models** with comparable accuracy
- **Lower memory** requirements
- **Faster inference** for deployment

### 3-Step Process

```bash
# 1. Train teacher (large, accurate model)
python scripts/train_fusion.py --config configs/teacher_large.yaml

# 2. Generate soft labels (one-time)
python data/preprocessing/soft_label_generator.py \
    --teacher_checkpoint checkpoints/teacher_best.pth \
    --video_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video \
    --audio_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio \
    --output_dir ./data/soft_labels

# 3. Train student (small, efficient model)
python scripts/train_fusion.py --config configs/student_small.yaml
```

## 📈 WandB Logging

### Automatic Tracking
- Training/validation loss and accuracy
- Learning rate scheduling
- Model complexity (params, GFLOPs)
- Per-class metrics
- Confusion matrices
- Attention visualizations

### Setup

```bash
# Login (one-time)
wandb login

# Configure in config file
logging:
  use_wandb: true
  wandb_project: my_experiments
```

## 📝 Configuration System

All models and training are configured via YAML files:

```yaml
# Example config structure
model:
  video_encoder: { type, output_dim, pretrained_path }
  audio_encoder: { type, output_dim, pretrained_path }
  fusion: { type, embed_dim, num_heads, num_layers }
  classifier: { num_classes, dropout }

data:
  data_dir, splits_file, batch_size, num_workers

training:
  epochs, learning_rate, optimizer, scheduler
  mixed_precision, gradient_clip
  checkpoint_dir, save_every

logging:
  use_wandb, wandb_project, log_every

hardware:
  device, gpu_id, seed
```

## 🔧 Complexity Analysis

### Automatic Calculation
- **Parameters**: Total, trainable, frozen
- **GFLOPs**: Computational complexity
- **Model Size**: Disk space required
- **Layer-wise Analysis**: Per-layer breakdown

### Usage

```python
from utils.complexity import analyze_model_complexity

analysis = analyze_model_complexity(
    model,
    video_shape=(1, 16, 3, 224, 224),
    audio_shape=(1, 64000)
)
# Prints detailed complexity report
```

## 📄 Report Generation

Generate comprehensive reports including:
- Architecture details
- Parameter count and GFLOPs
- Training history
- Evaluation results
- Confusion matrix

```bash
python scripts/generate_report.py \
    --checkpoint checkpoints/my_model/best.pth \
    --config configs/my_config.yaml \
    --output_dir reports/my_model
```

Outputs:
- `model_report.json` (machine-readable)
- `model_report.md` (human-readable)

## 🎯 Use Case Examples

### Academic Research
- Compare fusion methods
- Ablation studies
- Novel architecture experiments
- WandB for tracking

### Production Deployment
- Train large teacher model
- Use KD for efficient student
- Deploy student (MoViNet + MobileNet)
- ~10M params, fast inference

### Mobile/Edge
- MoViNet-A0 + MobileNet V2
- Simple concat fusion
- Mixed precision training
- <50MB model size

## 🔍 Key Files

### Essential Scripts
- `scripts/train_fusion.py` - Main training script
- `scripts/generate_report.py` - Report generation
- `data/preprocessing/video_preprocessing.py` - Video preprocessing
- `data/splits/split_utils.py` - Data splitting

### Core Modules
- `models/encoders/video_encoders.py` - 5 video encoders
- `models/encoders/audio_encoders.py` - 6 audio encoders
- `models/fusion/fusion_methods.py` - 5 fusion methods
- `utils/complexity/complexity_analysis.py` - Complexity metrics

### Configuration
- `configs/example_joint_crossattn.yaml` - Full fusion model
- `configs/example_kd_student.yaml` - KD configuration
- `requirements.txt` - Python dependencies
- `environment.yml` - Conda environment

## 📚 Documentation

- **README.md** - Overview and features
- **SETUP.md** - Installation and setup
- **USAGE.md** - Detailed usage examples
- **FRAMEWORK_SUMMARY.md** - This file

## ✨ Highlights

1. **Modular Design**: Easy to add custom encoders/fusion methods
2. **Flexible Configuration**: YAML-based config system
3. **Comprehensive Logging**: WandB integration
4. **Knowledge Distillation**: Offline mode for 2-3x speedup
5. **Complexity Analysis**: Automatic parameter/GFLOPs calculation
6. **Production Ready**: From research to deployment

## 🚀 Next Steps

1. **Setup**: Follow SETUP.md
2. **Preprocess Data**: Use video_preprocessing.py
3. **Configure**: Copy and modify example configs
4. **Train**: Run train_fusion.py
5. **Monitor**: Check WandB dashboard
6. **Evaluate**: Generate reports
7. **Deploy**: Use KD for efficient models

## 📧 Support

For questions and issues:
1. Check SETUP.md and USAGE.md
2. Review example configs
3. Check WandB logs
4. Review code documentation

## 🎉 Summary

This framework provides everything needed for audio-visual fusion:
- ✅ 5 video encoders + 6 audio encoders
- ✅ 5 fusion methods
- ✅ Knowledge distillation (online + offline)
- ✅ WandB logging
- ✅ Complexity analysis
- ✅ Comprehensive documentation
- ✅ Production-ready code

**Ready to train your multimodal models!** 🚀
