# Quick Start Guide

Train your first model in 5 minutes.

## ⚡ Ultra Quick Start

```bash
# 1. Activate environment
conda activate fusion_train_cu126
cd /home/hoang.pm/reorganized

# 2. Preprocess video and audio (run once)
python data/preprocessing/video_preprocessing.py \
    --input_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/video_dataset \
    --output_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video

python data/preprocessing/audio_preprocessing.py \
    --input_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/audio_dataset \
    --output_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio

# 3. Create fixed data splits (do ONCE for all experiments)
python data/splits/split_utils.py --create \
    --video_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video \
    --audio_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio \
    --output_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/splits \
    --seed 42 \
    --test_per_class 700 \
    --val_per_class 700

# 4. Update config with your data paths
nano configs/single_video.yaml
# Set:
#   video_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video
#   audio_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio
#   split_file: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/splits/splits.json

# 4. Train!
python scripts/train_single.py --config configs/single_video.yaml
```

Training will start immediately with WandB logging! 🎉

---

## 📋 Prerequisites

✅ **Environment**: `fusion_train_cu126` has all dependencies installed  
✅ **Data**: Video (.pkl) and audio (.npy) in **separate directories**  
✅ **GPU**: CUDA-capable GPU recommended (CPU training is very slow)

---

## 📁 Data Structure (Important!)

**Video and audio MUST be in separate folders:**

```
 /mnt/disk1/backup_user/hoang.pm/UFFIA_data/
├── fixed/
│   ├── processed_video/          # Preprocessed video (.pkl files)
│   ├── none/
│   │   ├── 1_video_1.pkl
│   │   ├── 2_video_1.pkl
│   │   └── ...
│   ├── weak/
│   ├── medium/
│   └── strong/
│   └── processed_audio/          # Preprocessed audio (.npy files)
│       ├── none/
│       │   ├── 1_audio_1.npy
│       │   ├── 2_audio_1.npy
│       │   └── ...
│       ├── weak/
│       ├── medium/
│       └── strong/
```

**File Matching**: Files are paired by naming convention
- Video: `XX_video_N.pkl` ↔ Audio: `XX_audio_N.npy`
- Example: `20_video_1.pkl` matches `20_audio_1.npy`

---

## 📊 Create Fixed Data Splits (Important!)

**Before running any experiments**, create fixed train/val/test splits to ensure fair comparison:

```bash
python data/splits/split_utils.py --create \
    --video_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video \
    --audio_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio \
    --output_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/splits \
    --seed 42 \
    --test_per_class 700 \
    --val_per_class 700
```

This creates:
```
/mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/splits/
├── splits.json      # Main split file (use this in configs)
├── train.json       # Training samples
├── val.json         # Validation samples  
└── test.json        # Test samples
```

**Why?** All your experiments (video-only, audio-only, fusion) will use the exact same data partitions, making results directly comparable.

---

## 🔍 Monitor & Validate Data

**Monitor preprocessing progress:**
```bash
python scripts/monitor_preprocessing.py --watch 60
```

**Check data leakage (duplicate IDs across classes/splits):**
```bash
python scripts/check_data_leakage.py
```

---

## 🎯 Choose Your Model

### Option 1: Video-Only (Recommended for First Run)

**Best for:** Camera-only setups, baselines

```bash
python scripts/train_single.py --config configs/single_video.yaml
```

**Expected results:**
- Time: 2-3 hours
- Accuracy: ~77%
- Parameters: 10M
- GFLOPs: 20

### Option 2: Audio-Only

**Best for:** Microphone-only setups, baselines

```bash
python scripts/train_single.py --config configs/single_audio.yaml
```

**Expected results:**
- Time: 2 hours
- Accuracy: ~67%
- Parameters: 5M
- GFLOPs: 3

### Option 3: Multimodal Fusion (Best Performance)

**Best for:** When both video and audio are available

```bash
python scripts/train_fusion.py --config configs/example_joint_crossattn.yaml
```

**Expected results:**
- Time: 3-4 hours
- Accuracy: ~87% ⭐
- Parameters: 45M
- GFLOPs: 48

---

## 📊 View Results

### WandB Dashboard

```bash
# First time: login to WandB
wandb login

# Training automatically logs to:
# https://wandb.ai/your_username/your_project
```

**Metrics tracked:**
- Train/val/test: accuracy, precision, recall, F1
- Confusion matrix
- Learning rate
- Model complexity (parameters, GFLOPs)

### Local Results

```bash
# View saved checkpoints
ls checkpoints/video_only_s3d/
# Output: best.pth, last.pth, test_results.json

# View test results
cat checkpoints/video_only_s3d/test_results.json
```

---

## ⚙️ Common Configurations

### Reduce GPU Memory

If you get CUDA Out of Memory errors:

```yaml
# In your config file
data:
  batch_size: 16  # Reduce from 32
```

### Quick Test Run

For testing the pipeline quickly:

```yaml
training:
  epochs: 5  # Reduce from 100
```

### Disable WandB

If you don't want to use WandB:

```yaml
logging:
  use_wandb: false
```

### Change Checkpoint Directory

```yaml
training:
  checkpoint_dir: ./checkpoints/my_experiment_name
```

---

## 🔄 Resume Training

If training was interrupted:

```bash
python scripts/train_single.py \
    --config configs/single_video.yaml \
    --resume checkpoints/video_only_s3d/last.pth
```

---

## 📁 If Your Data Is Not Ready

### Step 1: Preprocess Videos

```bash
python data/preprocessing/video_preprocessing.py \
    --input_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/video_dataset \
    --output_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video \
    --sampling_method uniform \
    --num_frames 16
```

**Sampling methods:**
- `uniform` - Evenly spaced frames (recommended)
- `random` - Random frames
- `consecutive` - Consecutive frames

### Step 2: Preprocess Audio

Convert raw audio to fixed-length `.npy` files:

```bash
python data/preprocessing/audio_preprocessing.py \
    --input_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/audio_dataset \
    --output_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio
```

Audio files should match video naming: `XX_audio_N.npy`

### Step 3: Update Config

Edit your config file:

```yaml
data:
  video_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video
  audio_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio
  seed: 42
  test_sample_per_class: 700  # Adjust based on your dataset size
```

### Step 4: Train

```bash
python scripts/train_single.py --config configs/single_video.yaml
```

---

## 🚀 Next Steps

### 1. Train All Baselines

```bash
# Video baseline
python scripts/train_single.py --config configs/single_video.yaml

# Audio baseline
python scripts/train_single.py --config configs/single_audio.yaml

# Fusion model
python scripts/train_fusion.py --config configs/example_joint_crossattn.yaml
```

### 2. Compare Results in WandB

All experiments are automatically logged to WandB for easy comparison!

### 3. Try Different Encoders

Edit config to use different encoders:

**Video encoders:**
- `s3d` - Default, good accuracy
- `x3d` - Efficient
- `movinet` - Mobile-friendly

**Audio encoders:**
- `pann_cnn10` - Default, balanced
- `pann_cnn14` - Higher accuracy
- `mobilenet` - Efficient

### 4. Try Different Fusion Methods

For fusion models:

```yaml
model:
  fusion:
    type: joint_cross_attention  # Options: concat, cross_attention, gated, joint_cross_attention, mbt
```

### 5. Generate Complexity Report

```bash
python scripts/train_generate_report.py \
    --checkpoint checkpoints/my_model/best.pth \
    --config configs/my_experiment.yaml \
    --output_dir reports/my_model
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```yaml
data:
  batch_size: 16  # or even 8
```

### Slow Training
```yaml
training:
  mixed_precision: true  # Enable if not already
data:
  num_workers: 2  # Reduce from 4
```

### Low Accuracy
- Check pretrained weights are loading correctly (look for "✓ Loaded pretrained" in logs)
- Train for more epochs (150-200)
- Try different encoder combinations
- Verify data quality and class balance

### Data Not Matching
- Verify file naming: `XX_video_N.pkl` ↔ `XX_audio_N.wav`
- Check class folder names: `none`, `weak`, `medium`, `strong`
- Look for warnings in dataset creation logs

### Import Errors
```bash
# Make sure you're in the right directory
cd /home/hoang.pm/reorganized

# Make sure environment is activated
conda activate fusion_train_cu126
```

---

## 📚 More Information

- **Complete Training Guide**: [TRAINING.md](TRAINING.md)
- **Installation & Setup**: [SETUP.md](SETUP.md)
- **Main Documentation**: [README.md](../README.md)

---

**You're all set!** 🎉 Start training and watch your models improve in WandB!
