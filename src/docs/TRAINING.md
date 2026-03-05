# Complete Training Guide

Comprehensive guide for training all model types: single modality, multimodal fusion, and knowledge distillation.

---

## 📖 Table of Contents

1. [Model Types](#model-types)
2. [Data Requirements](#data-requirements)
3. [Configuration](#configuration)
4. [Training Commands](#training-commands)
5. [Expected Results](#expected-results)
6. [Advanced Topics](#advanced-topics)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Model Types

### 1. Single Modality Models (Baselines)

Train models using only video or only audio.

**Use cases:**
- Building baselines for comparison
- Only one sensor available (camera-only or microphone-only)
- Ablation studies
- Resource-constrained deployments

**Video-Only:**
```bash
python scripts/train_single.py --config configs/single_video.yaml
```

**Audio-Only:**
```bash
python scripts/train_single.py --config configs/single_audio.yaml
```

### 2. Multimodal Fusion (Best Performance)

Combine video and audio for superior performance.

**Use cases:**
- Best possible accuracy
- Both sensors available
- Production deployments

```bash
python scripts/train_fusion.py --config configs/example_joint_crossattn.yaml
```

### 3. Knowledge Distillation (Efficient Models)

Train small, fast models that maintain high accuracy.

**Use cases:**
- Mobile/edge deployment
- Real-time inference
- Limited computational resources

```bash
# Offline KD (2-3x faster training)
python scripts/train_fusion.py --config configs/example_kd_student.yaml
```

---

## 📁 Data Requirements

### Directory Structure

**Critical**: Video and audio must be in **SEPARATE directories**:

```
/mnt/disk1/backup_user/hoang.pm/UFFIA_data/
└── fixed/
    ├── processed_video/          # Preprocessed video (.pkl files)
    │   ├── none/                 # Class name
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

### File Naming Convention

Files are matched by naming pattern:
- **Video**: `{id}_video_{index}.pkl` (e.g., `20_video_1.pkl`)
- **Audio**: `{id}_audio_{index}.npy` (e.g., `20_audio_1.npy`)

The dataset loader automatically pairs files with matching `{id}_{index}`.

### Data Splits

**Recommended: Use fixed splits** for fair comparison across experiments:

```bash
# Create fixed splits (do this ONCE before running any experiments)
python data/splits/split_utils.py --create \
    --video_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video \
    --audio_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio \
    --output_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/splits \
    --seed 42 \
    --test_per_class 700 \
    --val_per_class 700
```

This creates `data/splits/splits.json` containing train/val/test assignments.

**Configure in YAML:**
```yaml
data:
  split_file: ./data/splits/splits.json  # Fixed splits for fair comparison
  # seed and test_sample_per_class are ignored when split_file is provided
```

**Why fixed splits matter:**
- All experiments (video-only, audio-only, fusion, KD) use the exact same data
- Results are directly comparable
- No risk of data leakage between experiments

**Legacy mode** (not recommended for comparisons): If `split_file` is not provided, splits are generated dynamically:
```yaml
data:
  split_file: null  # Dynamic splitting
  seed: 42  # For reproducibility
  test_sample_per_class: 700  # Samples per class for test/val
```

---

## ⚙️ Configuration

### Basic Config Structure

```yaml
model:
  # Encoder configuration
  video_encoder:
    type: s3d
    output_dim: 1024
    pretrained_path: /mnt/disk1/backup_user/hoang.pm/pretrained_models/video/S3D_kinetics400.pth
  
  audio_encoder:
    type: pann_cnn10
    output_dim: 512
    pretrained_path: /mnt/disk1/backup_user/hoang.pm/pretrained_models/audio/Cnn10_mAP=0.380.pth
  
  # Fusion configuration (for fusion models only)
  fusion:
    type: joint_cross_attention
    embed_dim: 512
    num_heads: 8

data:
  video_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video
  audio_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio
  split_file: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/splits/splits.json
  seed: 42  # Only used if split_file not provided
  test_sample_per_class: 700  # Only used if split_file not provided
  batch_size: 32
  num_workers: 4
  audio_duration: 2.0
  sample_rate: 32000

training:
  epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.0001
  mixed_precision: true
  checkpoint_dir: ./checkpoints/my_model

logging:
  use_wandb: true
  wandb_project: my_experiments
```

### Change Video Encoder

```yaml
model:
  video_encoder:
    type: s3d       # Options: s3d, x3d, movinet, i3d, videomae
    output_dim: 1024
    pretrained_path: /mnt/disk1/backup_user/hoang.pm/pretrained_models/video/S3D_kinetics400.pth
```

**Available encoders:**
- `s3d` - Good accuracy, moderate speed (recommended)
- `x3d` - Efficient, fast inference
- `movinet` - Mobile-optimized, variants A0-A5
- `i3d` - High accuracy, slower
- `videomae` - Transformer-based, requires finetuning

### Change Audio Encoder

```yaml
model:
  audio_encoder:
    type: pann_cnn10  # Options: resnet18, resnet50, mobilenet, efficientnet, pann_cnn10, pann_cnn14
    output_dim: 512
    sample_rate: 32000
    pretrained_path: /mnt/disk1/backup_user/hoang.pm/pretrained_models/audio/Cnn10_mAP=0.380.pth
```

**Available encoders:**
- `pann_cnn10` - Balanced accuracy/speed (recommended)
- `pann_cnn14` - Higher accuracy, slower
- `resnet18` - Fast, lower accuracy
- `resnet50` - Better accuracy than ResNet-18
- `mobilenet` - Very efficient for mobile
- `efficientnet` - Good accuracy/efficiency trade-off

### Change Fusion Method

```yaml
model:
  fusion:
    type: joint_cross_attention  # Options below
    embed_dim: 512
    num_heads: 8
    num_layers: 2
    dropout: 0.1
```

**Available fusion methods:**
- `concat` - Simple concatenation + MLP (fastest, baseline)
- `cross_attention` - Bi-directional cross-modal attention
- `gated` - Learnable gating for modality weighting
- `joint_cross_attention` - Multi-layer attention (recommended, best accuracy)
- `mbt` - Multimodal Bottleneck Transformer (highest capacity)

### Adjust Training Parameters

```yaml
training:
  epochs: 100                  # More epochs = better convergence
  learning_rate: 0.0001        # Lower for fine-tuning, higher for training from scratch
  weight_decay: 0.0001         # Regularization
  optimizer: adamw             # Options: adamw, adam, sgd
  scheduler: cosine            # Options: cosine, plateau, none
  warmup_epochs: 5             # Gradual learning rate warmup
  gradient_clip: 1.0           # Prevent gradient explosion
  mixed_precision: true        # Faster training, less memory
  
  # Early stopping
  early_stopping:
    enabled: true
    patience: 15               # Stop if no improvement for N epochs
    min_delta: 0.001
```

### Adjust Data Loading

```yaml
data:
  batch_size: 32               # Reduce if OOM, increase for speed
  num_workers: 4               # Reduce if CPU bottleneck
  test_sample_per_class: 700   # Adjust based on dataset size
  audio_duration: 2.0          # Audio clip length in seconds
  sample_rate: 32000           # Audio sample rate
```

---

## 🚀 Training Commands

### Single Modality Training

**Video-Only:**
```bash
python scripts/train_single.py --config configs/single_video.yaml
```

**Audio-Only:**
```bash
python scripts/train_single.py --config configs/single_audio.yaml
```

**Run all video-only encoders (one WandB run each):**
```bash
python scripts/run_all_video_experiments.py \
    --video_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video_uniform \
    --audio_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio \
    --split_file /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/splits_uniform/splits.json \
    --wandb_project "Multimodalities for Aquaculture" \
    --wandb_entity "hoangpmh2406-vinuniversity" \
    --epochs 50
```

Video experiments train **all** supported models in sequence: **S3D**, **X3D** (XS/S/M/L), **I3D**, **MoViNet** (A0–A5), and **VideoMAE** (placeholder). Each uses its pretrained weights from `pretrained_models/video/` when `pretrained_path` is set. X3D/MoViNet require torchvision that exposes `x3d_*` and `movinet_*`; the script skips variants that are unavailable (install torchvision≥0.12 for full support).

**Resume training:**
```bash
python scripts/train_single.py \
    --config configs/single_video.yaml \
    --resume checkpoints/video_only_s3d/last.pth
```

### Fusion Model Training

**Basic fusion:**
```bash
python scripts/train_fusion.py --config configs/example_joint_crossattn.yaml
```

**With specific GPU:**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_fusion.py --config configs/example_joint_crossattn.yaml
```

**Resume fusion training:**
```bash
python scripts/train_fusion.py \
    --config configs/example_joint_crossattn.yaml \
    --resume checkpoints/fusion_model/last.pth
```

### Knowledge Distillation Training

**Step 1: Train teacher (one-time, or use existing best model)**
```bash
python scripts/train_fusion.py --config configs/teacher_large.yaml
```

**Step 2: Generate soft labels (one-time, for offline KD)**
```bash
python data/preprocessing/soft_label_generator.py \
    --teacher_checkpoint checkpoints/teacher/best.pth \
    --video_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video \
    --audio_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio \
    --output_dir data/soft_labels \
    --temperature 4.0 \
    --batch_size 32
```

This creates:
```
data/soft_labels/
├── train_soft_labels.pkl
├── val_soft_labels.pkl
└── test_soft_labels.pkl
```

**Step 3: Train student with offline KD**
```bash
python scripts/train_fusion.py --config configs/example_kd_student.yaml
```

---

## 📊 Expected Results

### Performance Comparison

| Model Type | Encoder(s) | Accuracy | Precision | Recall | F1 | Params | GFLOPs | Training Time |
|------------|------------|----------|-----------|--------|-------|--------|--------|---------------|
| Video-only | S3D | ~77% | ~76% | ~75% | ~75% | 10M | 20 | 2-3h |
| Audio-only | CNN10 | ~67% | ~65% | ~64% | ~64% | 5M | 3 | 2h |
| **Fusion** | S3D + CNN10 | **~87%** | **~86%** | **~85%** | **~85%** | 45M | 48 | 3-4h |
| KD Student | MoViNet-A0 + MobileNet | ~85% | ~84% | ~83% | ~83% | 9M | 6 | 2-3h |

### Factors Affecting Performance

**Positive factors:**
- Pretrained encoders (+5-10% accuracy)
- More training epochs (+2-5%)
- Better fusion methods (+3-7%)
- Data augmentation (+2-4%)
- Larger models (+3-5%)

**Negative factors:**
- Training from scratch (-10-15%)
- Too small batch size (-2-3%)
- Poor data quality (-10-20%)
- Class imbalance (-5-10%)

---

## 🎓 Advanced Topics

### Training Tips for Best Accuracy

1. **Use pretrained encoders**: Always use pretrained weights
2. **Train longer**: 150-200 epochs for best results
3. **Tune learning rate**: Start with 1e-4, reduce if training unstable
4. **Use mixed precision**: Faster and uses less memory
5. **Monitor validation**: Stop if overfitting (val loss increases)
6. **Ensemble models**: Train multiple models with different seeds

### Training Tips for Efficiency

1. **Use efficient encoders**: MoViNet-A0, MobileNet
2. **Knowledge distillation**: Train large teacher → compress to student
3. **Larger batch sizes**: With offline KD, use batch_size=64
4. **Simple fusion**: Concatenation is fastest
5. **Reduce precision**: Mixed precision training
6. **Optimize data loading**: SSD storage, more workers

### Hyperparameter Tuning

**Learning rate:**
```yaml
training:
  learning_rate: 0.0001  # Start here
  # Too high: training unstable, loss spikes
  # Too low: slow convergence, gets stuck
  # Try: 1e-3, 1e-4, 1e-5
```

**Batch size:**
```yaml
data:
  batch_size: 32  # Default
  # Larger: faster training, more stable gradients, needs more memory
  # Smaller: less memory, more noise in gradients
  # Try: 16, 32, 64
```

**Fusion embed dimension:**
```yaml
model:
  fusion:
    embed_dim: 512  # Default
    # Larger: more capacity, slower
    # Smaller: faster, less capacity
    # Try: 256, 512, 768, 1024
```

### Creating Custom Configs

```bash
# Copy existing config
cp configs/example_joint_crossattn.yaml configs/my_experiment.yaml

# Edit
nano configs/my_experiment.yaml

# Train
python scripts/train_fusion.py --config configs/my_experiment.yaml
```

### Monitoring Training

**WandB dashboard shows:**
- Real-time metrics (loss, accuracy, F1)
- Learning rate schedule
- Confusion matrices
- Model complexity
- System metrics (GPU usage, memory)

**Local logs:**
```bash
# View checkpoints
ls checkpoints/my_model/
# Output: best.pth, last.pth, test_results.json

# View test results
cat checkpoints/my_model/test_results.json
```

### Generating Reports

```bash
python scripts/train_generate_report.py \
    --checkpoint checkpoints/my_model/best.pth \
    --config configs/my_experiment.yaml \
    --output_dir reports/my_model
```

Generates:
- `model_report.json` - Machine-readable metrics
- `model_report.md` - Human-readable report
- Architecture diagram
- Complexity analysis (parameters, GFLOPs, size)
- Performance breakdown by class

---

## 🐛 Troubleshooting

### CUDA Out of Memory

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
```yaml
data:
  batch_size: 16  # Reduce from 32
  num_workers: 2  # Reduce from 4

training:
  mixed_precision: true  # Enable if not already
```

**Or use gradient accumulation:**
```yaml
training:
  batch_size: 16
  gradient_accumulation_steps: 2  # Effective batch size = 32
```

### Slow Training

**Solutions:**
1. Enable mixed precision:
```yaml
training:
  mixed_precision: true
```

2. Use faster encoders:
```yaml
model:
  video_encoder:
    type: x3d  # or movinet
  audio_encoder:
    type: mobilenet
```

3. Reduce data loading overhead:
```yaml
data:
  num_workers: 2
```

4. Use SSD for data storage

### Low Accuracy

**Check:**
1. Pretrained weights loaded? Look for "✓ Loaded pretrained" in logs
2. Learning rate too high? Try 1e-5
3. Training too short? Try 150-200 epochs
4. Data quality? Check for corrupted files
5. Class imbalance? Check class distribution in logs

**Try:**
- Different encoder combinations
- Better fusion method (joint_cross_attention)
- Longer training
- Data augmentation
- Lower learning rate

### Files Not Matching

**Symptoms:** "Warning: No matching audio for video..."

**Check:**
1. File naming: `XX_video_N.pkl` ↔ `XX_audio_N.npy`
2. Class folder names match: `none`, `weak`, `medium`, `strong`
3. Both directories specified correctly in config

**Debug:**
```bash
# List files
ls /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video/none/ | head
ls /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio/none/ | head

# Check naming pattern
```

### Training Stops/Crashes

**Solutions:**
1. Check disk space: `df -h`
2. Check memory: `free -h`
3. Check logs for errors
4. Resume from last checkpoint:
```bash
python scripts/train_fusion.py \
    --config configs/my_config.yaml \
    --resume checkpoints/my_model/last.pth
```

### Import Errors

**Symptoms:** `ModuleNotFoundError`

**Solutions:**
```bash
# Ensure correct directory
cd /home/hoang.pm/reorganized

# Ensure environment activated
conda activate fusion_train_cu126

# Check Python can find modules
python -c "import models.encoders; print('OK')"
```

---

## 📚 Next Steps

1. **Train baselines**: Video-only and audio-only models
2. **Train fusion**: Combine both modalities
3. **Compare in WandB**: Analyze results
4. **Try different configs**: Experiment with encoders and fusion
5. **Knowledge distillation**: Create efficient models
6. **Generate reports**: Document model complexity

**For more help:**
- [QUICK_START.md](QUICK_START.md) - Quick reference
- [SETUP.md](SETUP.md) - Installation guide
- [README.md](../README.md) - Overview

Good luck with your training! 🚀
