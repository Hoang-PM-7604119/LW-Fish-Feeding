# Multimodal Fusion Framework

Production-ready framework for audio-visual classification with multiple encoders, fusion methods, and knowledge distillation.

> **⚡ Quick Start**: See [docs/QUICK_START.md](docs/QUICK_START.md) to train your first model in 5 minutes!

## 🎯 Key Features

- **11 Encoders**: 5 video (S3D, X3D, MoViNet, I3D, VideoMAE) + 6 audio (ResNet-18/50, MobileNet V2, EfficientNet, PANN CNN10/14)
- **5 Fusion Methods**: Concatenation, Cross-Attention, Gated Fusion, Joint Cross-Attention, Multimodal Bottleneck Transformer (MBT)
- **3 Training Modes**: Single modality (video/audio only), multimodal fusion, knowledge distillation (online & offline)
- **Complete Logging**: WandB integration with train/val/test metrics, confusion matrices, and model complexity analysis
- **Separate Data Handling**: Video (.pkl) and audio (.npy) in separate directories

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[QUICK_START.md](docs/QUICK_START.md)** | Train your first model in 5 minutes |
| **[SETUP.md](docs/SETUP.md)** | Installation and environment setup |
| **[TRAINING.md](docs/TRAINING.md)** | Complete training guide for all model types |

## 🚀 Ultra Quick Start

```bash
# 1. Activate environment
conda activate fusion_train_cu126
cd /home/hoang.pm/reorganized

# 2. Verify data paths in config
# Edit config to set your data directories:
#   video_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video
#   audio_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio

# 3. Train (pick one)
python scripts/train_single.py --config configs/single_video.yaml        # Video-only baseline
python scripts/train_single.py --config configs/single_audio.yaml        # Audio-only baseline
python scripts/train_fusion.py --config configs/example_joint_crossattn.yaml  # Fusion model
```

## 📁 Data Structure

**Important**: Video and audio must be in **separate directories**:

```
/mnt/disk1/backup_user/hoang.pm/UFFIA_data/
└── fixed/
    ├── processed_video/          # Preprocessed video (.pkl files)
    │   ├── none/                 # Class folders
    │   │   ├── 1_video_1.pkl
    │   │   ├── 2_video_1.pkl
    │   │   └── ...
    │   ├── weak/
    │   ├── medium/
    │   └── strong/
    └── processed_audio/          # Preprocessed audio (.npy files)
        ├── none/
        │   ├── 1_audio_1.npy
        │   ├── 2_audio_1.npy
        │   └── ...
        ├── weak/
        ├── medium/
        └── strong/
```

Files are matched by naming: `XX_video_N.pkl` ↔ `XX_audio_N.npy`

## 📂 Project Structure

```
reorganized/
├── scripts/              # Training scripts
│   ├── train_fusion.py          # Multimodal fusion training
│   ├── train_single.py          # Single modality training
│   └── train_generate_report.py
├── configs/              # Configuration files (YAML)
│   ├── example_joint_crossattn.yaml  # Fusion model config
│   ├── example_kd_student.yaml       # Knowledge distillation config
│   ├── single_video.yaml             # Video-only config
│   └── single_audio.yaml             # Audio-only config
├── models/               # Model architectures
│   ├── encoders/                # Video and audio encoders
│   │   ├── video_encoders.py
│   │   └── audio_encoders.py
│   └── fusion/                  # Fusion methods
│       └── fusion_methods.py
├── data/                 # Data processing
│   ├── preprocessing/           # Video/audio preprocessing, soft labels
│   │   ├── video_preprocessing.py
│   │   ├── audio_preprocessing.py
│   │   └── soft_label_generator.py
│   ├── splits/                  # Data splitting utilities
│   │   └── split_utils.py
│   └── datasets/                # PyTorch datasets
│       └── multimodal_dataset.py
├── utils/                # Utilities
│   ├── metrics/                 # Accuracy, precision, recall, F1
│   ├── logging/                 # WandB integration
│   └── complexity/              # Parameters, GFLOPs calculation
├── docs/                 # Documentation
├── pretrained/           # (Optional) local weights
└── checkpoints/          # Saved models
```

## 📝 Training Examples

### Video-Only Model (Baseline)
```bash
python scripts/train_single.py --config configs/single_video.yaml
```
- **Time**: 2-3 hours
- **Accuracy**: ~77%
- **Parameters**: 10M

### Audio-Only Model (Baseline)
```bash
python scripts/train_single.py --config configs/single_audio.yaml
```
- **Time**: 2 hours
- **Accuracy**: ~67%
- **Parameters**: 5M

### Multimodal Fusion (Best Performance)
```bash
python scripts/train_fusion.py --config configs/example_joint_crossattn.yaml
```
- **Time**: 3-4 hours
- **Accuracy**: ~87%
- **Parameters**: 45M

### Knowledge Distillation (Efficient)
```bash
# Step 1: Train large teacher model (one-time)
python scripts/train_fusion.py --config configs/teacher_config.yaml

# Step 2: Generate soft labels (one-time)
python data/preprocessing/soft_label_generator.py \
    --teacher_checkpoint checkpoints/teacher/best.pth \
    --video_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video \
    --audio_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio \
    --output_dir data/soft_labels

# Step 3: Train efficient student model
python scripts/train_fusion.py --config configs/example_kd_student.yaml
```
- **Time**: 2-3 hours (2-3x faster than online KD)
- **Accuracy**: ~85%
- **Parameters**: 9M

### Generate Model Report
```bash
python scripts/train_generate_report.py \
    --checkpoint checkpoints/my_model/best.pth \
    --config configs/my_experiment.yaml \
    --output_dir reports/my_model
```

## 📊 Expected Results

| Model Type | Encoder(s) | Accuracy | Params | GFLOPs | Use Case |
|------------|------------|----------|--------|--------|----------|
| Video-only | S3D | ~77% | 10M | 20 | Camera-only setups |
| Audio-only | CNN10 | ~67% | 5M | 3 | Microphone-only setups |
| **Fusion** | S3D + CNN10 | **~87%** | 45M | 48 | Best accuracy |
| KD Student | MoViNet + MobileNet | ~85% | 9M | 6 | Mobile/edge deployment |

## 🎓 Model Components

### Video Encoders
- **S3D**: Spatiotemporal 3D CNN (Kinetics-400 pretrained)
- **X3D**: Efficient 3D CNN with expansion-compression
- **MoViNet**: Mobile Video Networks (A0-A5 variants)
- **I3D**: Inflated 3D ConvNet
- **Video MAE**: Masked autoencoder for video

### Audio Encoders
- **ResNet-18/50**: Deep residual networks
- **MobileNet V2**: Efficient mobile architecture
- **EfficientNet**: Compound scaling CNN
- **PANN CNN10/14**: Pretrained audio neural networks (AudioSet)

### Fusion Methods
- **Concat**: Simple feature concatenation + MLP
- **Cross-Attention**: Bi-directional cross-modal attention
- **Gated Fusion**: Learnable gating mechanism for modality weighting
- **Joint Cross-Attention**: Multi-layer self-attention + cross-attention
- **MBT**: Multimodal Bottleneck Transformer with shared latent space

## 📈 Logging & Metrics

**WandB tracks:**
- Train/val/test: accuracy, precision, recall, F1-score
- Confusion matrices
- Model complexity: parameters, GFLOPs, model size
- Learning rate schedule
- Loss components (for KD: hard loss, soft loss, feature loss)

**Saved artifacts:**
- `best.pth` - Best validation accuracy checkpoint
- `last.pth` - Last epoch checkpoint (for resuming)
- `test_results.json` - Detailed test evaluation results

## 🛠️ Requirements

**If using `fusion_train_cu126` environment:**
✅ All dependencies already installed

**For new setup:**
- Python 3.10+
- PyTorch 2.5.1+
- CUDA 12.6+ (for GPU training)
- See [SETUP.md](docs/SETUP.md) for complete installation

## 🔧 Configuration

Example config structure:

```yaml
model:
  video_encoder:
    type: s3d
    output_dim: 1024
    pretrained_path: /mnt/disk1/backup_user/hoang.pm/pretrained_models/video/S3D_kinetics400.pth
  
  audio_encoder:
    type: pann_cnn10
    output_dim: 512
    pretrained_path: /mnt/disk1/backup_user/hoang.pm/pretrained_models/audio/Cnn10_mAP=0.380.pth
  
  fusion:
    type: joint_cross_attention
    embed_dim: 512
    num_heads: 8

data:
  video_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video
  audio_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio
  seed: 42
  test_sample_per_class: 700
  batch_size: 32

training:
  epochs: 100
  learning_rate: 0.0001
  mixed_precision: true
  checkpoint_dir: ./checkpoints/my_model

logging:
  use_wandb: true
  wandb_project: fusion_experiments
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA OOM** | Reduce `batch_size` to 16 in config |
| **Slow training** | Enable `mixed_precision: true` |
| **Low accuracy** | Verify pretrained weights loaded correctly |
| **Data loading slow** | Reduce `num_workers` to 2 |
| **File not matched** | Check naming: `XX_video_N.pkl` ↔ `XX_audio_N.wav` |

## 🎯 Best Practices

**For Best Accuracy:**
- Use pretrained encoders (S3D, PANN CNN14)
- Train longer (150+ epochs)
- Use Joint Cross-Attention fusion
- Enable mixed precision training

**For Efficiency:**
- Use smaller encoders (MoViNet-A0, MobileNet)
- Apply knowledge distillation
- Use simple fusion (Concat)
- Larger batch sizes with offline KD

**For Production:**
1. Train large teacher model for best accuracy
2. Use knowledge distillation to create efficient student
3. Test student on target hardware
4. Monitor deployment with WandB

## 📄 Citation

If you use this framework, please cite the respective papers for the encoders and fusion methods you use.

## 📜 License

This project is provided for research and development purposes.

---

**Ready to start?** → [docs/QUICK_START.md](docs/QUICK_START.md) 🎉

**Need help?** Check [docs/TRAINING.md](docs/TRAINING.md) for complete training guide.
