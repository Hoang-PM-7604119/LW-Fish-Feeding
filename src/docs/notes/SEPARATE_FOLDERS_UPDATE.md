# Update: Separate Video and Audio Folders

## Overview

Updated the framework to correctly handle **separate video and audio directories**, matching the original implementation in `/home/hoang.pm/hoang/Fusion`.

## Key Changes

### 1. Data Structure

**Before (Incorrect):**
```
data/processed/
├── class1/
│   ├── video1.pkl
│   ├── video1.wav
│   └── ...
```

**After (Correct):**
```
/mnt/disk1/backup_user/hoang.pm/UFFIA_data/
└── fixed/
    ├── processed_video/      # Video files (.pkl)
    │   ├── none/
    │   │   ├── 1_video_1.pkl
    │   │   └── ...
    │   ├── weak/
    │   ├── medium/
    │   └── strong/
    └── processed_audio/      # Audio files (.npy)
        ├── none/
        │   ├── 1_audio_1.npy
        │   └── ...
        ├── weak/
        ├── medium/
        └── strong/
```

### 2. File Matching

Files are paired by naming convention:
- Video: `XX_video_N.pkl`
- Audio: `XX_audio_N.npy`
- Example: `20_video_1.pkl` ↔ `20_audio_1.npy`

### 3. Updated Files

#### Dataset (`data/datasets/multimodal_dataset.py`)
- Rewrote to match original implementation
- Takes separate `processed_video_path` and `audio_dataset_path`
- Implements `match_video_audio_pairs()` for file matching
- Uses `data_generator()` for train/val/test splits

#### Config Files
All configs updated with new data structure:
```yaml
data:
  video_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video
  audio_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio
  seed: 42
  test_sample_per_class: 700
  audio_duration: 2.0
  sample_rate: 32000
```

Updated files:
- `configs/example_joint_crossattn.yaml`
- `configs/example_kd_student.yaml`
- `configs/single_video.yaml`
- `configs/single_audio.yaml`

#### Training Scripts
- `scripts/train_fusion.py` - Updated to use `get_multimodal_dataloader()`
- `scripts/train_single.py` - Updated to use `get_multimodal_dataloader()`

Both now call:
```python
train_loader = get_multimodal_dataloader(
    processed_video_path=config['data']['video_dir'],
    audio_dataset_path=config['data']['audio_dir'],
    split='train',
    batch_size=config['data']['batch_size'],
    seed=config['data']['seed'],
    test_sample_per_class=config['data']['test_sample_per_class'],
    audio_sr=config['data']['sample_rate'],
    audio_duration=config['data']['audio_duration'],
    shuffle=True,
    num_workers=config['data']['num_workers']
)
```

#### Documentation
Updated to reflect separate folders:
- `docs/QUICK_START.md` - Data structure section
- `docs/TRAINING.md` - Configuration section
- `docs/SETUP.md` - Data preparation section

## Migration Guide

If you have existing data in combined folders:

### Option 1: Reorganize Data
```bash
# Split into separate folders
mkdir -p /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video \
  /mnt/disk1/backup_user/hoang.pm/UFFIA_data/audio_dataset \
  /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio

# Move video files
find data/processed -name "*_video_*.pkl" -exec mv {} /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video/ \;

# Move raw audio files
find data/processed -name "*_audio_*.wav" -exec mv {} /mnt/disk1/backup_user/hoang.pm/UFFIA_data/audio_dataset/ \;

# Preprocess audio to .npy
python data/preprocessing/audio_preprocessing.py \
  --input_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/audio_dataset \
  --output_dir /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio
```

### Option 2: Update Config
Simply point to your existing separate folders:
```yaml
data:
  video_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video
  audio_dir: /mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio
```

## Benefits

✅ Matches original implementation exactly  
✅ Cleaner separation of concerns  
✅ Flexible - video and audio can be on different drives  
✅ Supports partial datasets (some audio files missing)  
✅ Clear naming convention for pairing  

## Testing

To test the updated dataset:
```bash
cd /home/hoang.pm/reorganized
python data/datasets/multimodal_dataset.py
```

Update paths in the test section to your actual data directories.
