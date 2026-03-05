# ✅ Checkpoint & WandB Logging - Updated

## Summary

Updated both training scripts to:
1. **Save only best and last checkpoints** (removed periodic checkpoints)
2. **Enhanced WandB logging** with all metrics

## 🔧 Changes Made

### 1. Checkpoint Saving Strategy

**Before:**
- ✅ Saved best model
- ❌ Saved checkpoint every N epochs (cluttered checkpoint directory)
- ❌ No last checkpoint

**After:**
- ✅ Saves **best.pth** (best validation accuracy)
- ✅ Saves **last.pth** (last epoch, for resume)
- ❌ Removed periodic checkpoints (`save_every`)

### 2. WandB Logging

**Enhanced metrics logged:**
```python
wandb_logger.log_metrics({
    'train/loss': ...,
    'train/accuracy': ...,
    'train/precision': ...,
    'train/recall': ...,
    'train/f1': ...,
    'val/loss': ...,
    'val/accuracy': ...,
    'val/precision': ...,
    'val/recall': ...,
    'val/f1': ...,
    'learning_rate': ...
}, step=epoch)
```

**Features:**
- ✅ Grouped metrics (train/, val/)
- ✅ All classification metrics
- ✅ Learning rate tracking
- ✅ Per-epoch logging
- ✅ Proper namespacing

## 📁 Updated Files

### Scripts
1. **scripts/train_fusion.py**
   - Updated checkpoint logic (lines 234-285)
   - Enhanced WandB logging
   - Added last.pth saving

2. **scripts/train_single.py**
   - Updated checkpoint logic (lines 327-407)
   - Enhanced WandB logging
   - Added last.pth saving
   - Removed periodic checkpoint saving

### Configs
1. **configs/single_video.yaml**
   - Removed `save_every: 5`
   - Added comment about checkpoint strategy

2. **configs/single_audio.yaml**
   - Removed `save_every: 5`
   - Added comment about checkpoint strategy

3. **configs/example_joint_crossattn.yaml**
   - Removed `save_every: 5`
   - Added comment about checkpoint strategy

4. **configs/example_kd_student.yaml**
   - Removed `save_every: 5`
   - Added comment about checkpoint strategy

## 📊 Checkpoint Directory Structure

**After training:**
```
checkpoints/your_model/
├── best.pth       # Best model (highest val accuracy)
└── last.pth       # Last epoch (for resuming)
```

**Before (cluttered):**
```
checkpoints/your_model/
├── best.pth
├── checkpoint_epoch5.pth
├── checkpoint_epoch10.pth
├── checkpoint_epoch15.pth
├── checkpoint_epoch20.pth
└── ...
```

## 🎯 Benefits

### Cleaner Checkpoints
- ✅ Only 2 files per training run
- ✅ No clutter from periodic saves
- ✅ Easy to identify best and last
- ✅ Saves disk space

### Better WandB Logging
- ✅ All metrics tracked
- ✅ Grouped by train/val
- ✅ Easy to compare runs
- ✅ Better visualization

### Resume Training
```python
# Resume from last checkpoint
checkpoint = torch.load('checkpoints/my_model/last.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Use Best Model
```python
# Load best model for inference
checkpoint = torch.load('checkpoints/my_model/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Best val accuracy: {checkpoint['val_acc']:.4f}")
```

## 📈 WandB Dashboard

### Logged Metrics

**Training Metrics (per epoch):**
- `train/loss` - Training loss
- `train/accuracy` - Training accuracy
- `train/precision` - Training precision (macro)
- `train/recall` - Training recall (macro)
- `train/f1` - Training F1 score (macro)

**Validation Metrics (per epoch):**
- `val/loss` - Validation loss
- `val/accuracy` - Validation accuracy
- `val/precision` - Validation precision (macro)
- `val/recall` - Validation recall (macro)
- `val/f1` - Validation F1 score (macro)

**Test Metrics (after training):**
- `test/accuracy` - Test accuracy ⭐ NEW!
- `test/precision` - Test precision ⭐ NEW!
- `test/recall` - Test recall ⭐ NEW!
- `test/f1` - Test F1 score ⭐ NEW!
- `confusion_matrix` - Confusion matrix visualization ⭐ NEW!

**Training State:**
- `learning_rate` - Current learning rate

### View Dashboard

```bash
# After training starts
wandb login  # First time only

# View online
# Visit: https://wandb.ai/your_username/your_project
```

## 🚀 Usage

### Train with New Checkpoint Strategy

```bash
# Video-only
python scripts/train_single.py --config configs/single_video.yaml

# Audio-only
python scripts/train_single.py --config configs/single_audio.yaml

# Fusion
python scripts/train_fusion.py --config configs/example_joint_crossattn.yaml
```

**Results:**
```
checkpoints/video_only_s3d/
├── best.pth    # Saved when val accuracy improves
└── last.pth    # Saved every epoch (overwritten)
```

### Monitor on WandB

1. Training automatically logs to WandB (if `use_wandb: true`)
2. View metrics at: `https://wandb.ai/username/project`
3. Compare multiple runs
4. Download results as CSV

### Checkpoint Info

```python
import torch

# Check best checkpoint
best = torch.load('checkpoints/my_model/best.pth')
print(f"Best epoch: {best['epoch']}")
print(f"Best val accuracy: {best['val_acc']:.4f}")
print(f"Best val F1: {best['val_f1']:.4f}")

# Check last checkpoint
last = torch.load('checkpoints/my_model/last.pth')
print(f"Last epoch: {last['epoch']}")
print(f"Last val accuracy: {last['val_acc']:.4f}")
```

## 🔍 Verification

```bash
# Train for a few epochs
python scripts/train_single.py --config configs/single_video.yaml

# Check checkpoint directory
ls -lh checkpoints/video_only_s3d/
# Should show only: best.pth and last.pth

# Check WandB (if enabled)
# Visit dashboard to see all metrics
```

## ✅ Summary

**Checkpoint Strategy:**
- ✅ Saves **best.pth** (best validation accuracy)
- ✅ Saves **last.pth** (every epoch, for resume)
- ✅ No periodic checkpoints
- ✅ Clean and efficient

**WandB Logging:**
- ✅ All classification metrics
- ✅ Grouped by train/val
- ✅ Learning rate tracking
- ✅ Proper namespacing
- ✅ Easy comparison

**Both training scripts updated:**
- ✅ `scripts/train_fusion.py`
- ✅ `scripts/train_single.py`

**All configs updated:**
- ✅ Removed `save_every` parameter
- ✅ Added checkpoint strategy comment

**Ready to train with clean checkpointing and comprehensive WandB logging!** 🎉

---

Date: 2026-02-12
Status: ✅ Complete
