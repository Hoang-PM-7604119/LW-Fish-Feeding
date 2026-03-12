# Documentation Index

Complete documentation for the Multimodal Fusion Framework.

---

## 🚀 Quick Links

| What do you want to do? | Go to |
|--------------------------|-------|
| **Train my first model immediately** | [QUICK_START.md](QUICK_START.md) |
| **Set up the environment** | [SETUP.md](SETUP.md) |
| **Learn about all training options** | [TRAINING.md](TRAINING.md) |
| **See the project overview** | [../README.md](../README.md) |

---

## 📖 Main Documentation

### 1. [QUICK_START.md](QUICK_START.md)
**Train your first model in 5 minutes**

Perfect for:
- First-time users
- Quick testing
- Getting results fast

Contents:
- Ultra quick start (3 commands)
- Data structure explanation
- Basic configurations
- Viewing results
- Common troubleshooting

### 2. [SETUP.md](SETUP.md)
**Installation and environment setup**

Perfect for:
- New environment setup
- Data preparation
- Downloading pretrained models
- System configuration

Contents:
- Environment setup (conda/pip)
- Data directory structure
- File naming conventions
- Pretrained model downloads
- WandB setup
- Verification steps

### 3. [TRAINING.md](TRAINING.md)
**Complete training guide**

Perfect for:
- Understanding all model types
- Advanced configurations
- Hyperparameter tuning
- Production deployments

Contents:
- Single modality training
- Multimodal fusion training
- Knowledge distillation
- Configuration options
- Expected results
- Best practices
- Troubleshooting

---

## 📂 Additional Resources

### Project Overview
[../README.md](../README.md) - Main project README with overview, features, and examples

### Internal Notes
[notes/](notes/) - Internal documentation and change logs (for reference)

Key notes:
- [SEPARATE_FOLDERS_UPDATE.md](notes/SEPARATE_FOLDERS_UPDATE.md) - Update to separate video/audio folders
- [CHECKPOINT_WANDB_UPDATED.md](notes/CHECKPOINT_WANDB_UPDATED.md) - Checkpoint and logging updates
- [FRAMEWORK_SUMMARY.md](notes/FRAMEWORK_SUMMARY.md) - Quick framework reference

### Utility Scripts
- `scripts/monitor_preprocessing.py` - Watch preprocessing progress
- `scripts/check_data_leakage.py` - Detect duplicate IDs across classes/splits
- `scripts/run_all_video_experiments.py` - Run all video-only encoders with WandB

---

## 🎯 Recommended Reading Path

### For First-Time Users:
1. [../README.md](../README.md) - Get overview (5 min read)
2. [QUICK_START.md](QUICK_START.md) - Train first model (5 min setup)
3. [TRAINING.md](TRAINING.md) - Learn training options (when needed)

### For Production Deployment:
1. [SETUP.md](SETUP.md) - Set up properly
2. [TRAINING.md](TRAINING.md) - Understand all options
3. [QUICK_START.md](QUICK_START.md) - Quick reference

### For Troubleshooting:
1. Check [QUICK_START.md](QUICK_START.md) - Common issues section
2. Check [TRAINING.md](TRAINING.md) - Troubleshooting section  
3. Check [SETUP.md](SETUP.md) - Setup verification

---

## 📊 Training Workflow Overview

```
1. Setup
   ↓
   [SETUP.md] → Install environment + Prepare data
   
2. First Model
   ↓
   [QUICK_START.md] → Train video-only baseline (5 min)
   
3. Compare Models
   ↓
   [TRAINING.md] → Train audio + fusion models
   
4. Optimize
   ↓
   [TRAINING.md] → Knowledge distillation for efficiency
   
5. Deploy
   ↓
   [TRAINING.md] → Production best practices
```

---

## 🎓 Key Concepts

### Data Structure
Video and audio **must** be in separate directories:
```
/mnt/disk1/backup_user/hoang.pm/UFFIA_data/
└── fixed/
    ├── processed_video/    # .pkl files
    └── processed_audio/    # .npy files
```

Files matched by naming: `XX_video_N.pkl` ↔ `XX_audio_N.npy`

See: [SETUP.md](SETUP.md) - Data Preparation section

### Model Types
1. **Single Modality** - Video-only or audio-only
2. **Fusion** - Combine video + audio (best accuracy)
3. **Knowledge Distillation** - Efficient models

See: [TRAINING.md](TRAINING.md) - Model Types section

### Training Modes
- **Standard**: Train from scratch or with pretrained encoders
- **Resume**: Continue interrupted training
- **Knowledge Distillation**: Teacher→Student compression

See: [TRAINING.md](TRAINING.md) - Training Commands section

---

## 🔍 Finding Specific Topics

### Data-related
- **Data structure**: [SETUP.md](SETUP.md) - Data Preparation
- **File naming**: [SETUP.md](SETUP.md) - File Naming Convention
- **Preprocessing**: [SETUP.md](SETUP.md) - Step 1: Preprocess Videos

### Training-related
- **First training**: [QUICK_START.md](QUICK_START.md)
- **All training types**: [TRAINING.md](TRAINING.md) - Model Types
- **Configuration**: [TRAINING.md](TRAINING.md) - Configuration
- **Resume training**: [TRAINING.md](TRAINING.md) - Training Commands

### Models-related
- **Encoders**: [TRAINING.md](TRAINING.md) - Configuration section
- **Fusion methods**: [TRAINING.md](TRAINING.md) - Change Fusion Method
- **Performance**: [TRAINING.md](TRAINING.md) - Expected Results

### Troubleshooting
- **Quick fixes**: [QUICK_START.md](QUICK_START.md) - Troubleshooting
- **Detailed solutions**: [TRAINING.md](TRAINING.md) - Troubleshooting
- **Setup issues**: [SETUP.md](SETUP.md) - Troubleshooting

---

## ❓ Still Have Questions?

1. **Check the docs above** - Most questions are answered
2. **Look at example configs** - See `../configs/` for examples
3. **Check WandB logs** - Metrics and errors are logged there
4. **Review notes** - See `notes/` for implementation details

---

**Ready to start?** → [QUICK_START.md](QUICK_START.md) 🚀

**Need setup help?** → [SETUP.md](SETUP.md) 🛠️

**Want to learn everything?** → [TRAINING.md](TRAINING.md) 📚
