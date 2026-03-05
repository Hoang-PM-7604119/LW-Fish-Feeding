#!/usr/bin/env python
"""
Download MoViNet Pretrained Weights

Downloads pretrained MoViNet models from the MoViNet-pytorch repository:
https://github.com/Atze00/MoViNet-pytorch

Available models:
- MoViNet-A0 to A5 (Base and Stream variants)
- Pretrained on Kinetics-600

Usage:
    python download_movinet_weights.py --model a0 --output_dir ./pretrained_models
    python download_movinet_weights.py --all --output_dir ./pretrained_models
"""

import os
import sys
import argparse
import urllib.request
import hashlib
from pathlib import Path

# MoViNet weight URLs from the repository
# These are the direct download links for the converted PyTorch weights
MOVINET_WEIGHTS = {
    # Base models (standard 3D convolutions)
    'a0_base': {
        'url': 'https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA0_statedict_v3',
        'filename': 'movinet_a0_base.pt',
        'description': 'MoViNet-A0 Base (72.28% Top-1 on Kinetics-600)'
    },
    'a1_base': {
        'url': 'https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA1_statedict_v3',
        'filename': 'movinet_a1_base.pt',
        'description': 'MoViNet-A1 Base (76.69% Top-1 on Kinetics-600)'
    },
    'a2_base': {
        'url': 'https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA2_statedict_v3',
        'filename': 'movinet_a2_base.pt',
        'description': 'MoViNet-A2 Base (78.62% Top-1 on Kinetics-600)'
    },
    'a3_base': {
        'url': 'https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA3_statedict_v3',
        'filename': 'movinet_a3_base.pt',
        'description': 'MoViNet-A3 Base (81.79% Top-1 on Kinetics-600)'
    },
    'a4_base': {
        'url': 'https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA4_statedict_v3',
        'filename': 'movinet_a4_base.pt',
        'description': 'MoViNet-A4 Base (83.48% Top-1 on Kinetics-600)'
    },
    'a5_base': {
        'url': 'https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA5_statedict_v3',
        'filename': 'movinet_a5_base.pt',
        'description': 'MoViNet-A5 Base (84.27% Top-1 on Kinetics-600)'
    },
    # Stream models (with stream buffer for causal inference)
    'a0_stream': {
        'url': 'https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA0_stream_statedict_v3',
        'filename': 'movinet_a0_stream.pt',
        'description': 'MoViNet-A0 Stream (72.05% Top-1 on Kinetics-600)'
    },
    'a1_stream': {
        'url': 'https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA1_stream_statedict_v3',
        'filename': 'movinet_a1_stream.pt',
        'description': 'MoViNet-A1 Stream (76.45% Top-1 on Kinetics-600)'
    },
    'a2_stream': {
        'url': 'https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA2_stream_statedict_v3',
        'filename': 'movinet_a2_stream.pt',
        'description': 'MoViNet-A2 Stream (78.40% Top-1 on Kinetics-600)'
    },
}

# Recommended models for different use cases
RECOMMENDED = {
    'lightweight': 'a0_base',      # Smallest, fastest
    'balanced': 'a2_base',         # Good accuracy/speed tradeoff
    'accurate': 'a5_base',         # Best accuracy
    'streaming': 'a2_stream',      # Best for real-time streaming
}


def download_file(url, destination, description=""):
    """Download a file with progress indicator."""
    print(f"\nDownloading: {description}")
    print(f"  URL: {url}")
    print(f"  Destination: {destination}")
    
    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 // total_size)
                sys.stdout.write(f"\r  Progress: {percent}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
        print(f"\n  ✓ Downloaded successfully!")
        
        # Print file size
        size_mb = os.path.getsize(destination) / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")
        
        return True
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        return False


def download_model(model_key, output_dir):
    """Download a specific MoViNet model."""
    if model_key not in MOVINET_WEIGHTS:
        print(f"Error: Unknown model '{model_key}'")
        print(f"Available models: {', '.join(MOVINET_WEIGHTS.keys())}")
        return False
    
    model_info = MOVINET_WEIGHTS[model_key]
    output_path = Path(output_dir) / model_info['filename']
    
    # Skip if already exists
    if output_path.exists():
        print(f"\n✓ {model_info['filename']} already exists, skipping...")
        return True
    
    return download_file(
        url=model_info['url'],
        destination=str(output_path),
        description=model_info['description']
    )


def main():
    parser = argparse.ArgumentParser(
        description='Download MoViNet pretrained weights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download MoViNet-A0 Base model
  python download_movinet_weights.py --model a0_base

  # Download MoViNet-A2 (both base and stream)
  python download_movinet_weights.py --model a2_base --model a2_stream

  # Download all available models
  python download_movinet_weights.py --all

  # Download recommended models for your use case
  python download_movinet_weights.py --recommended lightweight
  python download_movinet_weights.py --recommended balanced
  python download_movinet_weights.py --recommended accurate

Available models:
  Base models (standard 3D convolutions):
    a0_base, a1_base, a2_base, a3_base, a4_base, a5_base
  
  Stream models (causal inference with stream buffer):
    a0_stream, a1_stream, a2_stream

Model performance on Kinetics-600:
  | Model          | Top-1 Acc | Input Shape     |
  |----------------|-----------|-----------------|
  | MoViNet-A0     | 72.28%    | 50 x 172 x 172  |
  | MoViNet-A1     | 76.69%    | 50 x 172 x 172  |
  | MoViNet-A2     | 78.62%    | 50 x 224 x 224  |
  | MoViNet-A3     | 81.79%    | 120 x 256 x 256 |
  | MoViNet-A4     | 83.48%    | 80 x 290 x 290  |
  | MoViNet-A5     | 84.27%    | 120 x 320 x 320 |
        """
    )
    
    parser.add_argument('--model', type=str, action='append', default=[],
                        help='Model(s) to download (can specify multiple)')
    parser.add_argument('--all', action='store_true',
                        help='Download all available models')
    parser.add_argument('--recommended', type=str, choices=['lightweight', 'balanced', 'accurate', 'streaming'],
                        help='Download recommended model for use case')
    parser.add_argument('--output_dir', type=str, default='./pretrained_models',
                        help='Output directory for weights (default: ./pretrained_models)')
    parser.add_argument('--list', action='store_true',
                        help='List available models and exit')
    
    args = parser.parse_args()
    
    # List models
    if args.list:
        print("\nAvailable MoViNet models:")
        print("-" * 60)
        for key, info in MOVINET_WEIGHTS.items():
            print(f"  {key:15} - {info['description']}")
        print("-" * 60)
        print("\nRecommended models:")
        for use_case, model in RECOMMENDED.items():
            print(f"  {use_case:15} -> {model}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which models to download
    models_to_download = []
    
    if args.all:
        models_to_download = list(MOVINET_WEIGHTS.keys())
    elif args.recommended:
        models_to_download = [RECOMMENDED[args.recommended]]
    elif args.model:
        models_to_download = args.model
    else:
        # Default: download A0 and A2 base models (most commonly used)
        models_to_download = ['a0_base', 'a2_base']
        print("No model specified. Downloading default models: a0_base, a2_base")
        print("Use --help to see all options.")
    
    print("\n" + "=" * 60)
    print("MoViNet Pretrained Weights Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Models to download: {', '.join(models_to_download)}")
    
    # Download models
    success_count = 0
    fail_count = 0
    
    for model_key in models_to_download:
        if download_model(model_key, output_dir):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Output directory: {output_dir.absolute()}")
    
    # Print usage instructions
    print("\n" + "=" * 60)
    print("Usage in your code:")
    print("=" * 60)
    print("""
# Install movinets package first:
pip install git+https://github.com/Atze00/MoViNet-pytorch.git

# Then use in your code:
from movinets import MoViNet
from movinets.config import _C
import torch

# Load model with pretrained weights (auto-downloads if needed)
model = MoViNet(_C.MODEL.MoViNetA0, causal=False, pretrained=True)

# Or load from your downloaded weights:
model = MoViNet(_C.MODEL.MoViNetA0, causal=False, pretrained=False)
state_dict = torch.load('pretrained_models/movinet_a0_base.pt')
model.load_state_dict(state_dict)
""")
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
