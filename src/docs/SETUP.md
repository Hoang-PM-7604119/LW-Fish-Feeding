chan# Setup Guide

Complete installation and setup instructions for the Multimodal Fusion Framework.

---

## ⚡ Quick Setup (If Using fusion_train_cu126)

**Already set up?** Just activate and use:

```bash
conda activate fusion_train_cu126
cd /home/hoang.pm/reorganized

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Start training
python scripts/train_single.py --config configs/single_video.yaml
```

**✅ Skip to [Data Preparation](#data-preparation) below.**

---

## 🆕 New Environment Setup

If you need to create a new environment from scratch:

### Prerequisites

- **CUDA**: 12.6+ for GPU support (11.8+ may work)
- **Disk Space**: 50GB+ (for data, models, checkpoints)
- **RAM**: 16GB+ recommended
- **GPU**: 8GB+ VRAM recommended (can run on CPU but very slow)

### Option 1: Conda (Recommended)
conda env create -f environment.yml

# Activate
conda activate fusion_framework

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Option 2: Pip

```bash
cd /home/hoang.pm/reorganized

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Verify Installation

```bash
# Test imports
python -c "
import torch
import torchvision
import librosa
import wandb
print('✓ All packages installed successfully')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"
```

---

## 📁 Data Preparation

### Critical: Separate Video and Audio Directories

**Video and audio MUST be in separate folders.** This is how the framework matches files.

```
/mnt/disk1/backup_user/hoang.pm/UFFIA_data/
├── video_dataset/                # Raw video (.mp4/.avi/.mov)
├── audio_dataset/                # Raw audio (.wav)
└── fixed/
    ├── processed_video/          # Preprocessed video (.pkl files)
    │   ├── none/
    │   ├── weak/
    │   ├── medium/
    │   └── strong/
    ├── processed_audio/          # Preprocessed audio (.npy files)
    │   ├── none/
    │   ├── weak/
    │   ├── medium/
    │   └── strong/
    └── splits/                   # Fixed train/val/test splits
        ├── splits.json
        ├── train.json
        ├── val.json
        └── test.json
```

### File Naming Convention

Files are **automatically paired** by naming pattern:

- **Video**: `{id}_video_{index}.pkl` (e.g., `20_video_1.pkl`)
- **Audio**: `{id}_audio_{index}.npy` (e.g., `20_audio_1.npy`)

**Examples of correct pairing:**
- `1_video_1.pkl` ↔ `1_audio_1.npy` ✅
- `20_video_5.pkl` ↔ `20_audio_5.npy` ✅
- `100_video_2.pkl` ↔ `100_audio_2.npy` ✅

### Step 1: Preprocess Videos

Convert raw videos to preprocessed pickle files:

```bash
python data/preprocessing/video_preprocessing.py \
    --input_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/video_dataset \
    --output_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video \
    --sampling_method uniform \
    --num_frames 16
```

**Parameters:**
- `--input_dir`: Directory with raw video files (.mp4, .avi, etc.)
- `--output_dir`: Where to save preprocessed .pkl files
- `--sampling_method`: How to sample frames
  - `uniform` (recommended) - Evenly spaced frames
  - `random` - Random frames
  - `consecutive` - Consecutive frames from start
- `--num_frames`: Number of frames per video (default: 16)

**Output:**
```
/mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video/
├── none/
│   ├── 1_video_1.pkl
│   └── ...
├── weak/
├── medium/
└── strong/
```

Each `.pkl` file contains:
```python
{
    'video_form': torch.Tensor,  # Shape: [16, 3, 224, 224]
    'metadata': dict             # Optional metadata
}
```

### Step 2: Preprocess Audio

Convert raw audio to fixed-length `.npy` files:

```bash
python data/preprocessing/audio_preprocessing.py \
    --input_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/audio_dataset \
    --output_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio \
    --sample_rate 32000 \
    --duration 2.0
```

**Audio preprocessing behavior:**
- Input format: `.wav` files
- Output format: `.npy` files (float32)
- Sample rate: resampled to 32kHz
- Duration: trimmed/padded to 2 seconds
- Channels: converted to mono

### Step 3: Create Fixed Data Splits (RECOMMENDED)

**For fair comparison across experiments**, create fixed train/val/test splits:

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
├── splits.json      # Main split file with metadata
├── train.json       # Training samples
├── val.json         # Validation samples  
└── test.json        # Test samples
```

**Why use fixed splits?**
- All experiments use exactly the same data partitions
- Results are directly comparable across different models
- Reproducible experiments regardless of code changes

### Step 4: Update Config Files

Edit your config to specify both paths and the split file:

```yaml
data:
  video_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video
  audio_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio
  
  # Fixed splits for fair comparison (RECOMMENDED)
  split_file: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/splits/splits.json
  
  # These are ignored when split_file is provided
  seed: 42
  test_sample_per_class: 700
  
  batch_size: 32
  num_workers: 4
  audio_duration: 2.0                # Audio duration in seconds
  sample_rate: 32000                 # Audio sample rate
```

**Important:** Use absolute paths if files are not in the default location:
```yaml
data:
  video_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video
  audio_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio
  split_file: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/splits/splits.json
```

### Step 5: Verify Data

Check that files are correctly paired:

```bash
# List some video files
ls /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video/none/ | head -n 5

# List corresponding audio files
ls /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio/none/ | head -n 5

# Should have matching numbers!
```

Test the dataset loader:

```bash
python -c "
from data.datasets.multimodal_dataset import get_multimodal_dataloader

loader = get_multimodal_dataloader(
    processed_video_path='/mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video',
    audio_dataset_path='/mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio',
    split='train',
    batch_size=4,
    seed=42,
    test_sample_per_class=700,
    shuffle=True,
    num_workers=2
)

batch = next(iter(loader))
print(f'✓ Dataset test passed!')
print(f'Video shape: {batch[\"video\"].shape}')
print(f'Audio shape: {batch[\"audio\"].shape}')
print(f'Labels: {batch[\"label\"]}')
"
```

---

## 🎯 Download Pretrained Models

Use the downloader script (recommended):
```bash
python download_pretrained_models.py --all
```

Default output directory:
```
/mnt/disk1/backup_user/hoang.pm/pretrained_models/
├── video/
│   └── S3D_kinetics400.pth
└── audio/
    ├── Cnn10_mAP=0.380.pth
    └── Cnn14_mAP=0.431.pth
```

Most torchvision-based encoders will download automatically when `pretrained: true`.
S3D and PANNs use the files above via `pretrained_path`.

---

## 🚀 Start Training

### Create Your Config

```bash
# Copy an example config
cp configs/example_joint_crossattn.yaml configs/my_experiment.yaml

# Edit paths and parameters
nano configs/my_experiment.yaml
```

**Minimal config:**
```yaml
model:
  video_encoder:
    type: s3d
    output_dim: 1024
  audio_encoder:
    type: pann_cnn10
    output_dim: 512
  fusion:
    type: joint_cross_attention
    embed_dim: 512

data:
  video_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video
  audio_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio
  batch_size: 32

training:
  epochs: 100
  checkpoint_dir: ./checkpoints/my_model

logging:
  use_wandb: true
```

### Train Your Model

```bash
python scripts/train_fusion.py --config configs/my_experiment.yaml
```

---

## 📊 WandB Setup

### First Time Setup

```bash
# Login to Weights & Biases
wandb login

# Enter your API key from: https://wandb.ai/authorize
```

### Configure WandB in Config

```yaml
logging:
  use_wandb: true
  wandb_project: my_fusion_experiments  # Project name
  wandb_entity: null                     # Your username (optional)
  log_every: 10                          # Log every N batches
```

### Disable WandB (Optional)

If you don't want to use WandB:

```yaml
logging:
  use_wandb: false
```

Metrics will still be logged locally.

---

## 📈 Generate Model Reports

After training, generate comprehensive reports:

```bash
python scripts/train_generate_report.py \
    --checkpoint checkpoints/my_model/best.pth \
    --config configs/my_experiment.yaml \
    --output_dir reports/my_model
```

**Generates:**
- `model_report.json` - Machine-readable metrics
- `model_report.md` - Human-readable report
- Architecture details
- Complexity analysis (parameters, GFLOPs, model size)
- Performance breakdown by class

---

## 🐛 Troubleshooting

### CUDA Out of Memory

**Solution:**
```yaml
data:
  batch_size: 16  # Reduce from 32
```

Or use CPU (very slow):
```yaml
hardware:
  device: cpu
```

### Slow Training

**Solutions:**
1. Enable mixed precision:
```yaml
training:
  mixed_precision: true
```

2. Use SSD for data storage

3. Reduce data loading overhead:
```yaml
data:
  num_workers: 2  # Reduce from 4
```

### Import Errors

```bash
# Ensure correct directory
cd /home/hoang.pm/reorganized

# Ensure environment activated
conda activate fusion_train_cu126

# Test imports
python -c "import models.encoders; print('OK')"
```

### Data Loading Errors

**"Warning: No matching audio for video..."**

Check file naming:
```bash
# Video files should be: XX_video_N.pkl
ls /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video/none/ | head

# Audio files should be: XX_audio_N.npy
ls /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio/none/ | head
```

### WandB Not Logging

```bash
# Re-login
wandb login

# Check config
grep -A5 "logging:" configs/my_experiment.yaml

# Ensure use_wandb: true
```

---

## 🔧 Advanced Configuration

### Multi-GPU Training

```yaml
hardware:
  device: cuda
  gpu_ids: [0, 1]  # Use GPUs 0 and 1
```

Or via environment variable:
```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_fusion.py --config configs/my_config.yaml
```

### Custom Data Splits

**Recommended: Use fixed splits** for fair comparison across experiments:

```bash
# Create fixed splits (do this once)
python data/splits/split_utils.py --create \
    --video_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video \
    --audio_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio \
    --output_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/splits \
    --seed 42 \
    --test_per_class 700 \
    --val_per_class 700
```

Then reference in config:
```yaml
data:
  split_file: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/splits/splits.json
```

**Legacy mode (dynamic splitting):** If no `split_file` is provided, splits are generated automatically based on `seed` and `test_sample_per_class`. This is not recommended for comparing multiple experiments.

```yaml
data:
  split_file: null  # or omit this line
  seed: 42
  test_sample_per_class: 500  # Smaller dataset
```

### Resume Training

```bash
python scripts/train_fusion.py \
    --config configs/my_experiment.yaml \
    --resume checkpoints/my_model/last.pth
```

---

## 📚 Next Steps

1. ✅ Verify installation and data setup
2. ✅ Train video-only baseline - [QUICK_START.md](QUICK_START.md)
3. ✅ Train audio-only baseline
4. ✅ Train fusion model
5. ✅ Experiment with different encoders and fusion methods - [TRAINING.md](TRAINING.md)
6. ✅ Try knowledge distillation for efficient models

**Setup complete!** 🎉

Ready to train? See [QUICK_START.md](QUICK_START.md) to train your first model in 5 minutes!
