# Multimodal Fusion Framework

Production-ready framework for audio‑visual classification with multiple encoders, fusion methods, and knowledge distillation.

**Want to run it now?** See `docs/QUICK_START.md` for a step‑by‑step guide.

## Key Features

- **Encoders**: Multiple video (S3D, X3D, MoViNet, I3D, VideoMAE) and audio (ResNet‑18/50, MobileNet V2, EfficientNet, PANN CNN10/14) backbones
- **Fusion**: Concatenation, cross‑attention, gated fusion, joint cross‑attention, and Multimodal Bottleneck Transformer (MBT)
- **Training modes**: Single modality (video or audio), multimodal fusion, and knowledge distillation
- **Logging**: Weights & Biases integration and local JSON reports
- **Data layout**: Video (`.pkl`) and audio (`.npy`) stored in separate folders and paired by filename

## Ultra Quick Start

```bash
# 1. Activate your environment
conda activate <your_env>      # e.g. fusion_train_cu126

# 2. Go to the source folder
cd /path/to/LW-Fish-Feeding/src

# 3. Train (pick one)
python scripts/train_single.py --config configs/single_video.yaml        # Video-only baseline
python scripts/train_single.py --config configs/single_audio.yaml        # Audio-only baseline
python scripts/train_fusion.py --config configs/example_joint_crossattn.yaml  # Video+audio fusion
```

Before training, make sure the config points to your processed data:

```yaml
data:
  video_dir: /path/to/data/processed_video
  audio_dir: /path/to/data/processed_audio
  split_file: /path/to/data/splits/splits.json   # optional but recommended
```

## Data Structure (Simplified)

Video and audio must live in separate directories and use a matching name pattern:

```
/path/to/data/
└── fixed/
    ├── processed_video/
    │   ├── none/
    │   ├── weak/
    │   ├── medium/
    │   └── strong/
    └── processed_audio/
        ├── none/
        ├── weak/
        ├── medium/
        └── strong/
```

Files are paired by name, e.g. `20_video_1.pkl` ↔ `20_audio_1.npy`.

## Project Layout (this folder)

```
src/
├── scripts/              # Entry-point scripts
│   ├── train_fusion.py
│   ├── train_single.py
│   ├── generate_report.py
│   ├── create_fixed_splits.py
│   ├── monitor_preprocessing.py
│   └── run_all_*.py
├── configs/              # YAML configs for all experiments
├── models/               # Encoders and fusion modules
├── data/                 # Datasets, preprocessing, and splits
├── utils/                # Logging, metrics, complexity analysis
├── docs/                 # Documentation
└── checkpoints/          # Saved models (created at runtime)
```

## Where to Go Next

- **Get started quickly**: `docs/QUICK_START.md`
- **Install and prepare data**: `docs/SETUP.md`
- **See all training options**: `docs/TRAINING.md`
- **Browse doc index**: `docs/README.md`
