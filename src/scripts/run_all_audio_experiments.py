#!/usr/bin/env python3
"""
Comprehensive Audio Model Training Script

Trains all available audio model types with:
- Fixed train/val/test splits (700 samples per class for val/test)
- 50 epochs per model
- WandB logging for experiment tracking

Audio Model Types:
1. ResNet18 - 2D ResNet on mel-spectrogram
2. ResNet50 - Larger ResNet variant
3. MobileNetV2 - Efficient mobile architecture
4. EfficientNet-B0 - Compound scaling CNN
5. PANN CNN10 - Pretrained Audio Neural Network (10 layers)
6. PANN CNN14 - Pretrained Audio Neural Network (14 layers)

Usage:
    python scripts/run_all_audio_experiments.py

Or run specific models:
    python scripts/run_all_audio_experiments.py --models resnet18 mobilenet pann_cnn10
"""

import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import yaml
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


AUDIO_MODELS = {
    'resnet18': {
        'type': 'resnet18',
        'output_dim': 512,
        'description': 'ResNet-18 on mel-spectrogram'
    },
    'resnet50': {
        'type': 'resnet50',
        'output_dim': 2048,
        'description': 'ResNet-50 on mel-spectrogram'
    },
    'mobilenet': {
        'type': 'mobilenet',
        'output_dim': 1280,
        'description': 'MobileNetV2 - efficient mobile architecture'
    },
    'efficientnet': {
        'type': 'efficientnet',
        'output_dim': 1280,
        'description': 'EfficientNet-B0 - compound scaling CNN',
        'kwargs': {'variant': 'b0'}
    },
    'pann_cnn10': {
        'type': 'pann_cnn10',
        'output_dim': 512,
        'description': 'PANN CNN10 - Pretrained Audio Neural Network',
        'pretrained_path': '/mnt/disk1/backup_user/hoang.pm/pretrained_models/audio/Cnn10_mAP=0.380.pth'
    },
    'pann_cnn14': {
        'type': 'pann_cnn14',
        'output_dim': 2048,
        'description': 'PANN CNN14 - Larger Pretrained Audio Neural Network',
        'pretrained_path': '/mnt/disk1/backup_user/hoang.pm/pretrained_models/audio/Cnn14_mAP=0.431.pth'
    }
}


def create_fixed_splits(video_dir, audio_dir, output_dir, val_per_class=700, test_per_class=700, seed=42):
    """Create fixed train/val/test splits."""
    print("\n" + "=" * 80)
    print("Step 1: Creating Fixed Data Splits")
    print("=" * 80)
    
    split_file = os.path.join(output_dir, 'splits', 'splits.json')
    
    if os.path.exists(split_file):
        print(f"✓ Split file already exists: {split_file}")
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        print(f"  Train: {split_data['statistics']['train']} samples")
        print(f"  Val: {split_data['statistics']['val']} samples")
        print(f"  Test: {split_data['statistics']['test']} samples")
        return split_file
    
    script_path = Path(__file__).parent / 'create_fixed_splits.py'
    
    cmd = [
        sys.executable, str(script_path),
        '--video_dir', video_dir,
        '--audio_dir', audio_dir,
        '--output_dir', os.path.join(output_dir, 'splits'),
        '--seed', str(seed),
        '--val_per_class', str(val_per_class),
        '--test_per_class', str(test_per_class)
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Failed to create splits:")
        print(result.stderr)
        sys.exit(1)
    
    print(result.stdout)
    return split_file


def generate_config(model_name, model_config, base_config):
    """Generate configuration for a specific audio model."""
    config = {
        'model': {
            'modality': 'audio',
            'audio_encoder': {
                'type': model_config['type'],
                'output_dim': model_config['output_dim'],
                'sample_rate': base_config['sample_rate'],
            },
            'classifier': {
                'num_classes': 4,
                'dropout': base_config['dropout']
            }
        },
        'data': {
            'video_dir': base_config['video_dir'],
            'audio_dir': base_config['audio_dir'],
            'split_file': base_config['split_file'],
            'seed': base_config['seed'],
            'test_sample_per_class': 700,
            'batch_size': base_config['batch_size'],
            'num_workers': base_config['num_workers'],
            'audio_duration': base_config['audio_duration'],
            'sample_rate': base_config['sample_rate'],
            'class_names': ['none', 'weak', 'medium', 'strong']
        },
        'training': {
            'epochs': base_config['epochs'],
            'learning_rate': base_config['learning_rate'],
            'weight_decay': base_config['weight_decay'],
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'warmup_epochs': 5,
            'gradient_clip': 1.0,
            'mixed_precision': True,
            'checkpoint_dir': os.path.join(base_config['output_dir'], 'checkpoints', f'audio_{model_name}'),
            'early_stopping': {
                'enabled': True,
                'patience': 15,
                'min_delta': 0.001
            }
        },
        'logging': {
            'use_wandb': base_config['use_wandb'],
            'wandb_project': base_config['wandb_project'],
            'wandb_entity': base_config.get('wandb_entity'),
            'log_every': 10
        },
        'evaluation': {
            'test_batch_size': 64,
            'save_predictions': True,
            'save_confusion_matrix': True
        },
        'hardware': {
            'device': 'cuda',
            'gpu_id': 0,
            'deterministic': True,
            'seed': base_config['seed']
        }
    }
    
    if 'pretrained_path' in model_config:
        config['model']['audio_encoder']['pretrained_path'] = model_config['pretrained_path']
    
    if 'kwargs' in model_config:
        config['model']['audio_encoder']['kwargs'] = model_config['kwargs']
    else:
        config['model']['audio_encoder']['kwargs'] = {}
    
    return config


def train_model(config_path, model_name):
    """Train a single audio model."""
    print(f"\n{'='*80}")
    print(f"Training: {model_name.upper()}")
    print(f"Config: {config_path}")
    print(f"{'='*80}\n")
    
    script_path = Path(__file__).parent / 'train_single.py'
    
    cmd = [sys.executable, str(script_path), '--config', config_path]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Audio Model Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models:
  resnet18     - ResNet-18 on mel-spectrogram (512-dim)
  resnet50     - ResNet-50 on mel-spectrogram (2048-dim)
  mobilenet    - MobileNetV2 efficient architecture (1280-dim)
  efficientnet - EfficientNet-B0 compound scaling (1280-dim)
  pann_cnn10   - Pretrained Audio Neural Network 10-layer (512-dim)
  pann_cnn14   - Pretrained Audio Neural Network 14-layer (2048-dim)

Examples:
  # Train all models with default settings
  python scripts/run_all_audio_experiments.py \\
      --video_dir ./data/processed_video \\
      --audio_dir ./data/audio_dataset

  # Train specific models
  python scripts/run_all_audio_experiments.py \\
      --models resnet18 mobilenet pann_cnn10 \\
      --epochs 50

  # Full configuration
  python scripts/run_all_audio_experiments.py \\
      --video_dir ./data/processed_video \\
      --audio_dir ./data/audio_dataset \\
      --output_dir ./experiments/audio_v2 \\
      --wandb_project fish_audio_v2 \\
      --epochs 50 \\
      --batch_size 32 \\
      --learning_rate 0.0001
        """
    )
    
    parser.add_argument('--video_dir', type=str,
                        default='/mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video',
                        help='Directory containing video files')
    parser.add_argument('--audio_dir', type=str,
                        default='/mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio',
                        help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/disk1/backup_user/hoang.pm/UFFIA_data/experiments/audio_single',
                        help='Output directory for experiments')
    
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        choices=list(AUDIO_MODELS.keys()) + ['all'],
                        help='Models to train (default: all)')
    
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay (default: 0.0001)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    
    parser.add_argument('--val_per_class', type=int, default=700,
                        help='Validation samples per class (default: 700)')
    parser.add_argument('--test_per_class', type=int, default=700,
                        help='Test samples per class (default: 700)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    parser.add_argument('--audio_duration', type=float, default=2.0,
                        help='Audio duration in seconds (default: 2.0)')
    parser.add_argument('--sample_rate', type=int, default=32000,
                        help='Audio sample rate (default: 32000)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='Use WandB for logging (default: True)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable WandB logging')
    parser.add_argument('--wandb_project', type=str, default='fish_audio_comprehensive',
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='WandB entity/team name')
    
    parser.add_argument('--skip_splits', action='store_true',
                        help='Skip split creation (use existing splits)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Generate configs only, do not train')
    
    args = parser.parse_args()
    
    if args.no_wandb:
        args.use_wandb = False
    
    if args.models is None or 'all' in args.models:
        models_to_train = list(AUDIO_MODELS.keys())
    else:
        models_to_train = args.models
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE AUDIO MODEL TRAINING")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nModels to train ({len(models_to_train)}):")
    for model in models_to_train:
        print(f"  - {model}: {AUDIO_MODELS[model]['description']}")
    print(f"\nTraining settings:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Val/Test samples per class: {args.val_per_class}/{args.test_per_class}")
    print(f"  WandB: {'Enabled' if args.use_wandb else 'Disabled'}")
    if args.use_wandb:
        print(f"  WandB project: {args.wandb_project}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'configs').mkdir(exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'splits').mkdir(exist_ok=True)
    
    if not args.skip_splits:
        split_file = create_fixed_splits(
            args.video_dir,
            args.audio_dir,
            args.output_dir,
            val_per_class=args.val_per_class,
            test_per_class=args.test_per_class,
            seed=args.seed
        )
    else:
        split_file = os.path.join(args.output_dir, 'splits', 'splits.json')
        if not os.path.exists(split_file):
            print(f"✗ Split file not found: {split_file}")
            print("  Run without --skip_splits to create splits first")
            sys.exit(1)
    
    base_config = {
        'video_dir': args.video_dir,
        'audio_dir': args.audio_dir,
        'split_file': split_file,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'audio_duration': args.audio_duration,
        'sample_rate': args.sample_rate,
        'num_workers': args.num_workers,
        'seed': args.seed,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        'wandb_entity': args.wandb_entity
    }
    
    print("\n" + "=" * 80)
    print("Step 2: Generating Configuration Files")
    print("=" * 80)
    
    config_paths = {}
    for model_name in models_to_train:
        model_config = AUDIO_MODELS[model_name]
        config = generate_config(model_name, model_config, base_config)
        
        config_path = output_dir / 'configs' / f'audio_{model_name}.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        config_paths[model_name] = str(config_path)
        print(f"✓ Generated: {config_path}")
    
    experiment_info = {
        'timestamp': datetime.now().isoformat(),
        'models': models_to_train,
        'base_config': base_config,
        'config_files': config_paths,
        'split_file': split_file
    }
    
    with open(output_dir / 'experiment_info.json', 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN - Configs generated, skipping training")
        print("=" * 80)
        print(f"\nTo train manually, run:")
        for model_name, config_path in config_paths.items():
            print(f"  python scripts/train_single.py --config {config_path}")
        return
    
    print("\n" + "=" * 80)
    print("Step 3: Training Models")
    print("=" * 80)
    
    results = {}
    for i, model_name in enumerate(models_to_train, 1):
        print(f"\n[{i}/{len(models_to_train)}] Training {model_name}...")
        
        try:
            return_code = train_model(config_paths[model_name], model_name)
            
            if return_code == 0:
                results_path = output_dir / 'checkpoints' / f'audio_{model_name}' / 'test_results.json'
                if results_path.exists():
                    with open(results_path, 'r') as f:
                        test_results = json.load(f)
                    results[model_name] = {
                        'status': 'success',
                        'test_accuracy': test_results.get('test_accuracy'),
                        'test_f1': test_results.get('test_f1')
                    }
                    print(f"✓ {model_name}: Accuracy={test_results.get('test_accuracy', 0):.4f}, "
                          f"F1={test_results.get('test_f1', 0):.4f}")
                else:
                    results[model_name] = {'status': 'success', 'note': 'No results file found'}
            else:
                results[model_name] = {'status': 'failed', 'return_code': return_code}
                print(f"✗ {model_name}: Training failed with return code {return_code}")
        
        except Exception as e:
            results[model_name] = {'status': 'error', 'error': str(e)}
            print(f"✗ {model_name}: Error - {e}")
    
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    summary = []
    for model_name, result in results.items():
        if result['status'] == 'success' and 'test_accuracy' in result:
            summary.append({
                'model': model_name,
                'accuracy': result['test_accuracy'],
                'f1': result['test_f1']
            })
            print(f"  {model_name:15s}: Acc={result['test_accuracy']:.4f}, F1={result['test_f1']:.4f}")
        else:
            print(f"  {model_name:15s}: {result['status']}")
    
    if summary:
        summary.sort(key=lambda x: x['accuracy'], reverse=True)
        print(f"\nBest Model by Accuracy: {summary[0]['model']} ({summary[0]['accuracy']:.4f})")
        summary.sort(key=lambda x: x['f1'], reverse=True)
        print(f"Best Model by F1: {summary[0]['model']} ({summary[0]['f1']:.4f})")
    
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'models_trained': len(models_to_train),
        'results': results,
        'summary': summary
    }
    
    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir / 'final_results.json'}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
