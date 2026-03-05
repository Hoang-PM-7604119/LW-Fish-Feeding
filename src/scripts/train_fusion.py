#!/usr/bin/env python3
"""
Training script for fusion models.

Usage:
    python scripts/train_fusion.py --config configs/example_joint_crossattn.yaml
"""

import argparse
import os
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.encoders import get_video_encoder, get_audio_encoder
from models.fusion import get_fusion_method
from data.datasets.multimodal_dataset import get_multimodal_dataloader
from utils.metrics import calculate_metrics, AverageMeter, print_metrics
from utils.logging import init_wandb
from utils.complexity import analyze_model_complexity


class MultimodalFusionModel(nn.Module):
    """Complete fusion model with encoders, fusion, and classifier."""
    
    def __init__(self, config):
        super().__init__()
        
        # Video encoder
        self.video_encoder = get_video_encoder(
            encoder_type=config['model']['video_encoder']['type'],
            output_dim=config['model']['video_encoder']['output_dim'],
            **config['model']['video_encoder'].get('kwargs', {})
        )
        
        # Audio encoder
        self.audio_encoder = get_audio_encoder(
            encoder_type=config['model']['audio_encoder']['type'],
            output_dim=config['model']['audio_encoder']['output_dim'],
            **config['model']['audio_encoder'].get('kwargs', {})
        )
        
        # Fusion module
        self.fusion = get_fusion_method(
            fusion_type=config['model']['fusion']['type'],
            video_dim=config['model']['video_encoder']['output_dim'],
            audio_dim=config['model']['audio_encoder']['output_dim'],
            embed_dim=config['model']['fusion']['embed_dim'],
            **config['model']['fusion'].get('kwargs', {})
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config['model']['fusion']['embed_dim'], 
                     config['model']['fusion']['embed_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['model']['classifier']['dropout']),
            nn.Linear(config['model']['fusion']['embed_dim'] // 2,
                     config['model']['classifier']['num_classes'])
        )
    
    def forward(self, video, audio):
        v_feat = self.video_encoder(video)
        a_feat = self.audio_encoder(audio)
        fused, _ = self.fusion(v_feat, a_feat)
        logits = self.classifier(fused)
        return logits


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    for batch_idx, batch in enumerate(dataloader):
        video = batch['video'].to(device)
        audio = batch['audio'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(video, audio)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        loss_meter.update(loss.item(), video.size(0))
        
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = loss_meter.avg
    
    return metrics


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    loss_meter = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(video, audio)
            loss = criterion(logits, labels)
            
            loss_meter.update(loss.item(), video.size(0))
            
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = loss_meter.avg
    
    return metrics


def test(model, dataloader, device):
    """Test model and return detailed metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(video, audio)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    
    return metrics, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description='Train fusion model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n{'='*80}")
    print(f"Training Fusion Model")
    print(f"{'='*80}")
    print(f"Config: {args.config}")
    
    # Set device
    device = torch.device(config['hardware']['device'])
    print(f"Device: {device}")
    
    # Set seed
    torch.manual_seed(config['hardware']['seed'])
    
    # Create model
    print(f"\nBuilding model...")
    model = MultimodalFusionModel(config).to(device)
    
    # Analyze complexity
    analyze_model_complexity(model, device=str(device))
    
    # Create dataloaders with separate video and audio directories
    print(f"\nCreating dataloaders...")
    print(f"Video directory: {config['data']['video_dir']}")
    print(f"Audio directory: {config['data']['audio_dir']}")
    
    # Get split file path (recommended for fair comparison)
    split_file = config['data'].get('split_file', None)
    if split_file:
        print(f"Split file: {split_file} (fixed splits for fair comparison)")
    else:
        print(f"Split mode: Dynamic (seed={config['data']['seed']})")
        print(f"  TIP: Use fixed splits for fair comparison across experiments:")
        print(f"       python scripts/create_fixed_splits.py --help")
    
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
        drop_last=True,
        num_workers=config['data']['num_workers'],
        split_file=split_file
    )
    
    val_loader = get_multimodal_dataloader(
        processed_video_path=config['data']['video_dir'],
        audio_dataset_path=config['data']['audio_dir'],
        split='val',
        batch_size=config['data']['batch_size'],
        seed=config['data']['seed'],
        test_sample_per_class=config['data']['test_sample_per_class'],
        audio_sr=config['data']['sample_rate'],
        audio_duration=config['data']['audio_duration'],
        shuffle=False,
        drop_last=False,
        num_workers=config['data']['num_workers'],
        split_file=split_file
    )
    
    test_loader = get_multimodal_dataloader(
        processed_video_path=config['data']['video_dir'],
        audio_dataset_path=config['data']['audio_dir'],
        split='test',
        batch_size=config['data']['batch_size'],
        seed=config['data']['seed'],
        test_sample_per_class=config['data']['test_sample_per_class'],
        audio_sr=config['data']['sample_rate'],
        audio_duration=config['data']['audio_duration'],
        shuffle=False,
        drop_last=False,
        num_workers=config['data']['num_workers'],
        split_file=split_file
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Initialize WandB
    wandb_logger = None
    if config['logging']['use_wandb']:
        wandb_logger = init_wandb(
            project=config['logging']['wandb_project'],
            config=config
        )
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"Starting Training")
    print(f"{'='*80}\n")
    
    best_val_acc = 0.0
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config['training']['epochs']):
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        print(f"{'-'*80}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        
        # Log metrics to WandB
        if wandb_logger:
            wandb_logger.log_metrics({
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'train/precision': train_metrics['precision'],
                'train/recall': train_metrics['recall'],
                'train/f1': train_metrics['f1'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'val/f1': val_metrics['f1'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            checkpoint_path = checkpoint_dir / 'best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_f1': val_metrics['f1'],
                'config': config
            }, checkpoint_path)
            print(f"✓ Saved best model (acc: {best_val_acc:.4f}, f1: {val_metrics['f1']:.4f})")
        
        # Save last checkpoint (overwrite each epoch)
        last_checkpoint_path = checkpoint_dir / 'last.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['f1'],
            'config': config
        }, last_checkpoint_path)
        
        scheduler.step()
        print()
    
    # Evaluate on test set
    print(f"\n{'='*80}")
    print(f"Evaluating on Test Set")
    print(f"{'='*80}")
    
    # Load best model
    best_checkpoint = torch.load(checkpoint_dir / 'best.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Test evaluation
    test_metrics, test_labels, test_preds = test(model, test_loader, device)
    
    print(f"\nTest Results:")
    print_metrics(test_metrics, prefix="")
    
    # Log test metrics to WandB
    if wandb_logger:
        wandb_logger.log_metrics({
            'test/accuracy': test_metrics['accuracy'],
            'test/precision': test_metrics['precision'],
            'test/recall': test_metrics['recall'],
            'test/f1': test_metrics['f1'],
        })
        
        # Log confusion matrix
        wandb_logger.log_confusion_matrix(
            y_true=test_labels,
            y_pred=test_preds,
            class_names=config['data']['class_names']
        )
    
    # Save test metrics to file
    import json
    test_results_path = checkpoint_dir / 'test_results.json'
    with open(test_results_path, 'w') as f:
        json.dump({
            'test_accuracy': float(test_metrics['accuracy']),
            'test_precision': float(test_metrics['precision']),
            'test_recall': float(test_metrics['recall']),
            'test_f1': float(test_metrics['f1']),
            'confusion_matrix': test_metrics['confusion_matrix'],
            'best_val_accuracy': float(best_val_acc),
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"\nCheckpoints saved to: {checkpoint_dir}")
    print(f"  - best.pth: Best model (val_acc: {best_val_acc:.4f})")
    print(f"  - last.pth: Last epoch checkpoint")
    print(f"  - test_results.json: Test evaluation results")
    print(f"{'='*80}\n")
    
    if wandb_logger:
        wandb_logger.finish()


if __name__ == '__main__':
    main()
