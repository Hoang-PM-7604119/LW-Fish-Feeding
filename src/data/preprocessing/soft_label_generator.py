"""
Soft label generator for knowledge distillation.

Generates soft labels (teacher predictions) offline for faster KD training.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def generate_soft_labels(
    teacher_model: nn.Module,
    dataloader: DataLoader,
    output_dir: str,
    temperature: float = 4.0,
    device: str = 'cuda',
    save_features: bool = True
):
    """
    Generate soft labels from teacher model.
    
    Args:
        teacher_model: Trained teacher model
        dataloader: DataLoader for the dataset
        output_dir: Directory to save soft labels
        temperature: Temperature for softening logits
        device: Device to run inference on
        save_features: Whether to save intermediate features
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    
    print(f"\n{'='*80}")
    print(f"Generating Soft Labels")
    print(f"{'='*80}")
    print(f"  Teacher model: {teacher_model.__class__.__name__}")
    print(f"  Temperature: {temperature}")
    print(f"  Output directory: {output_dir}")
    print(f"  Device: {device}")
    print(f"{'='*80}\n")
    
    soft_labels_data = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating soft labels")):
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].cpu().numpy()
            
            # Get teacher outputs
            if save_features:
                # Assume model has return_features option
                try:
                    logits, features = teacher_model(video, audio, return_features=True)
                except:
                    # Fallback if model doesn't support return_features
                    logits = teacher_model(video, audio)
                    features = None
            else:
                logits = teacher_model(video, audio)
                features = None
            
            # Soften logits
            soft_labels = F.softmax(logits / temperature, dim=1).cpu().numpy()
            
            # Save for each sample in batch
            for i in range(len(labels)):
                sample_id = f"{batch_idx}_{i}"
                
                data = {
                    'soft_labels': soft_labels[i],
                    'hard_labels': labels[i],
                    'temperature': temperature
                }
                
                if features is not None:
                    data['features'] = {
                        k: v[i].cpu().numpy() if isinstance(v, torch.Tensor) else v
                        for k, v in features.items()
                    }
                
                soft_labels_data[sample_id] = data
    
    # Save all soft labels
    output_file = output_path / 'soft_labels.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(soft_labels_data, f)
    
    print(f"\n{'='*80}")
    print(f"Soft Label Generation Complete")
    print(f"{'='*80}")
    print(f"  Total samples: {len(soft_labels_data)}")
    print(f"  Saved to: {output_file}")
    print(f"{'='*80}\n")
    
    return soft_labels_data


def load_soft_labels(soft_labels_path: str) -> Dict:
    """
    Load precomputed soft labels.
    
    Args:
        soft_labels_path: Path to soft labels file
        
    Returns:
        soft_labels_data: Dictionary of soft labels
    """
    with open(soft_labels_path, 'rb') as f:
        soft_labels_data = pickle.load(f)
    
    print(f"✓ Loaded {len(soft_labels_data)} soft labels from {soft_labels_path}")
    return soft_labels_data


if __name__ == '__main__':
    """Test soft label generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate soft labels for KD')
    parser.add_argument('--teacher_checkpoint', type=str, required=True,
                       help='Path to teacher model checkpoint')
    parser.add_argument('--data_dir', type=str,
                       help='Data directory (legacy: single root)')
    parser.add_argument('--video_dir', type=str,
                       help='Processed video directory')
    parser.add_argument('--audio_dir', type=str,
                       help='Processed audio directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for soft labels')
    parser.add_argument('--temperature', type=float, default=4.0,
                       help='Temperature for softening logits')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    if not args.data_dir and not (args.video_dir and args.audio_dir):
        print("Error: provide --data_dir or both --video_dir and --audio_dir")
        raise SystemExit(1)

    print("Soft label generation script")
    print("Note: This is a template. Implement model loading and dataset creation.")
