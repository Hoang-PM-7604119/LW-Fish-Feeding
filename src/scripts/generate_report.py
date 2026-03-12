#!/usr/bin/env python3
"""
Generate comprehensive model report including:
- Architecture details
- Parameter count and GFLOPs
- Training metrics
- Evaluation results
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.complexity import analyze_model_complexity, print_complexity_analysis


def generate_report(checkpoint_path, config_path, output_dir):
    """Generate comprehensive model report."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Generating Model Report")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create report dictionary
    report = {
        'model': {
            'architecture': {
                'video_encoder': config['model']['video_encoder']['type'],
                'audio_encoder': config['model']['audio_encoder']['type'],
                'fusion_method': config['model']['fusion']['type'],
            },
            'hyperparameters': config['model']
        },
        'training': {
            'epochs_trained': checkpoint.get('epoch', 'N/A'),
            'best_val_acc': checkpoint.get('val_acc', 'N/A'),
            'config': config.get('training', {})
        }
    }
    
    # Model complexity analysis (if model can be reconstructed)
    print("Model Complexity Analysis:")
    print(f"{'-'*80}")
    
    # Note: This is a simplified version - you'd need to reconstruct the model
    # from the checkpoint to get accurate complexity metrics
    complexity_summary = {
        'note': 'Reconstruct model from checkpoint for accurate metrics',
        'checkpoint_size_mb': Path(checkpoint_path).stat().st_size / (1024**2)
    }
    
    if 'model_state_dict' in checkpoint:
        # Count parameters from state dict
        total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
        complexity_summary['total_parameters'] = int(total_params)
        complexity_summary['total_parameters_millions'] = total_params / 1e6
        
        print(f"  Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    print(f"  Checkpoint Size: {complexity_summary['checkpoint_size_mb']:.2f} MB")
    
    report['complexity'] = complexity_summary
    
    # Save report as JSON
    report_file = output_path / 'model_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to {report_file}")
    
    # Generate markdown report
    md_report = generate_markdown_report(report)
    md_file = output_path / 'model_report.md'
    with open(md_file, 'w') as f:
        f.write(md_report)
    
    print(f"✓ Markdown report saved to {md_file}")
    
    print(f"\n{'='*80}")
    print(f"Report Generation Complete!")
    print(f"{'='*80}\n")


def generate_markdown_report(report):
    """Generate markdown formatted report."""
    
    md = "# Model Report\n\n"
    
    # Architecture
    md += "## Architecture\n\n"
    md += f"- **Video Encoder**: {report['model']['architecture']['video_encoder']}\n"
    md += f"- **Audio Encoder**: {report['model']['architecture']['audio_encoder']}\n"
    md += f"- **Fusion Method**: {report['model']['architecture']['fusion_method']}\n"
    md += "\n"
    
    # Complexity
    md += "## Model Complexity\n\n"
    if 'total_parameters_millions' in report['complexity']:
        md += f"- **Parameters**: {report['complexity']['total_parameters_millions']:.2f}M\n"
    md += f"- **Checkpoint Size**: {report['complexity']['checkpoint_size_mb']:.2f} MB\n"
    md += "\n"
    
    # Training
    md += "## Training\n\n"
    md += f"- **Epochs**: {report['training']['epochs_trained']}\n"
    md += f"- **Best Validation Accuracy**: {report['training']['best_val_acc']:.4f}\n"
    md += "\n"
    
    # Hyperparameters
    md += "## Hyperparameters\n\n"
    md += "```yaml\n"
    md += yaml.dump(report['training']['config'], default_flow_style=False)
    md += "```\n"
    
    return md


def main():
    parser = argparse.ArgumentParser(description='Generate model report')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./reports',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    generate_report(args.checkpoint, args.config, args.output_dir)


if __name__ == '__main__':
    main()
