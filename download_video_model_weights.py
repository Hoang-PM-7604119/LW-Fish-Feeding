#!/usr/bin/env python
"""
Download Pretrained Video Model Weights

Downloads pretrained weights for various video classification models:
- S3D (Kinetics-400) - via torchvision
- X3D (Kinetics-400) - via pytorchvideo/torch.hub
- I3D (Kinetics-400) - direct download
- Video ViT / TimeSformer / Video Swin (various)

Usage:
    python download_video_model_weights.py --model s3d --output_dir ./pretrained_models
    python download_video_model_weights.py --all --output_dir ./pretrained_models
    python download_video_model_weights.py --list
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path

# Video model weight URLs - Direct download links
VIDEO_MODEL_WEIGHTS = {
    # S3D - Separable 3D CNN (torchvision official weights)
    's3d_kinetics400': {
        'url': 'https://download.pytorch.org/models/s3d-d76dad2f.pth',
        'filename': 'S3D_kinetics400.pth',
        'description': 'S3D pretrained on Kinetics-400 (torchvision official)',
        'source': 'torchvision.models.video.s3d'
    },
    
    # I3D - Inflated 3D ConvNet
    'i3d_rgb_kinetics400': {
        'url': 'https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt',
        'filename': 'I3D_rgb_imagenet_kinetics.pt',
        'description': 'I3D RGB pretrained on ImageNet+Kinetics',
        'source': 'https://github.com/piergiaj/pytorch-i3d'
    },
    'i3d_flow_kinetics400': {
        'url': 'https://github.com/piergiaj/pytorch-i3d/raw/master/models/flow_imagenet.pt',
        'filename': 'I3D_flow_imagenet_kinetics.pt',
        'description': 'I3D Flow pretrained on ImageNet+Kinetics',
        'source': 'https://github.com/piergiaj/pytorch-i3d'
    },
    'i3d_rgb_charades': {
        'url': 'https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_charades.pt',
        'filename': 'I3D_rgb_charades.pt',
        'description': 'I3D RGB finetuned on Charades',
        'source': 'https://github.com/piergiaj/pytorch-i3d'
    },
    
    # X3D - Expanded 3D (from facebookresearch pytorchvideo)
    'x3d_xs': {
        'url': 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_XS.pyth',
        'filename': 'X3D_XS_kinetics400.pyth',
        'description': 'X3D-XS pretrained on Kinetics-400 (Extra Small)',
        'source': 'https://github.com/facebookresearch/pytorchvideo'
    },
    'x3d_s': {
        'url': 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_S.pyth',
        'filename': 'X3D_S_kinetics400.pyth',
        'description': 'X3D-S pretrained on Kinetics-400 (Small)',
        'source': 'https://github.com/facebookresearch/pytorchvideo'
    },
    'x3d_m': {
        'url': 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_M.pyth',
        'filename': 'X3D_M_kinetics400.pyth',
        'description': 'X3D-M pretrained on Kinetics-400 (Medium)',
        'source': 'https://github.com/facebookresearch/pytorchvideo'
    },
    'x3d_l': {
        'url': 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_L.pyth',
        'filename': 'X3D_L_kinetics400.pyth',
        'description': 'X3D-L pretrained on Kinetics-400 (Large)',
        'source': 'https://github.com/facebookresearch/pytorchvideo'
    },
    
    # SlowFast
    'slowfast_r50': {
        'url': 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST_8x8_R50.pyth',
        'filename': 'SlowFast_8x8_R50_kinetics400.pyth',
        'description': 'SlowFast R50 8x8 pretrained on Kinetics-400',
        'source': 'https://github.com/facebookresearch/pytorchvideo'
    },
    'slowfast_r101': {
        'url': 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST_16x8_R101_50_50.pyth',
        'filename': 'SlowFast_16x8_R101_kinetics400.pyth',
        'description': 'SlowFast R101 16x8 pretrained on Kinetics-400',
        'source': 'https://github.com/facebookresearch/pytorchvideo'
    },
    
    # Slow (R50 backbone)
    'slow_r50': {
        'url': 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW_8x8_R50.pyth',
        'filename': 'Slow_8x8_R50_kinetics400.pyth',
        'description': 'Slow R50 8x8 pretrained on Kinetics-400',
        'source': 'https://github.com/facebookresearch/pytorchvideo'
    },
    
    # C2D (2D ConvNet baseline)
    'c2d_r50': {
        'url': 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/C2D_8x8_R50.pyth',
        'filename': 'C2D_8x8_R50_kinetics400.pyth',
        'description': 'C2D R50 8x8 pretrained on Kinetics-400',
        'source': 'https://github.com/facebookresearch/pytorchvideo'
    },
    
    # I3D from pytorchvideo
    'i3d_r50_pytorchvideo': {
        'url': 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/I3D_8x8_R50.pyth',
        'filename': 'I3D_8x8_R50_kinetics400.pyth',
        'description': 'I3D R50 8x8 pretrained on Kinetics-400 (pytorchvideo)',
        'source': 'https://github.com/facebookresearch/pytorchvideo'
    },
    
    # CSN (Channel-Separated Networks)
    'csn_r101': {
        'url': 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/CSN_32x2_R101.pyth',
        'filename': 'CSN_32x2_R101_kinetics400.pyth',
        'description': 'CSN R101 32x2 pretrained on Kinetics-400',
        'source': 'https://github.com/facebookresearch/pytorchvideo'
    },
    
    # R(2+1)D
    'r2plus1d_r50': {
        'url': 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/R2PLUS1D_16x4_R50.pyth',
        'filename': 'R2plus1D_16x4_R50_kinetics400.pyth',
        'description': 'R(2+1)D R50 16x4 pretrained on Kinetics-400',
        'source': 'https://github.com/facebookresearch/pytorchvideo'
    },
    
    # Video Swin Transformer
    'video_swin_tiny_k400': {
        'url': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth',
        'filename': 'VideoSwin_tiny_k400.pth',
        'description': 'Video Swin Transformer Tiny on Kinetics-400',
        'source': 'https://github.com/SwinTransformer/Video-Swin-Transformer'
    },
    'video_swin_small_k400': {
        'url': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_small_patch244_window877_kinetics400_1k.pth',
        'filename': 'VideoSwin_small_k400.pth',
        'description': 'Video Swin Transformer Small on Kinetics-400',
        'source': 'https://github.com/SwinTransformer/Video-Swin-Transformer'
    },
    'video_swin_base_k400': {
        'url': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_1k.pth',
        'filename': 'VideoSwin_base_k400.pth',
        'description': 'Video Swin Transformer Base on Kinetics-400',
        'source': 'https://github.com/SwinTransformer/Video-Swin-Transformer'
    },
    'video_swin_base_k600': {
        'url': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics600_22k.pth',
        'filename': 'VideoSwin_base_k600.pth',
        'description': 'Video Swin Transformer Base on Kinetics-600',
        'source': 'https://github.com/SwinTransformer/Video-Swin-Transformer'
    },
    
    # MViT (Multiscale Vision Transformers) v1
    'mvit_base_16x4': {
        'url': 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/MVIT_B_16x4.pyth',
        'filename': 'MViT_B_16x4_kinetics400.pyth',
        'description': 'MViT Base 16x4 pretrained on Kinetics-400',
        'source': 'https://github.com/facebookresearch/pytorchvideo'
    },
    'mvit_base_32x3': {
        'url': 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/MVIT_B_32x3.pyth',
        'filename': 'MViT_B_32x3_kinetics400.pyth',
        'description': 'MViT Base 32x3 pretrained on Kinetics-400',
        'source': 'https://github.com/facebookresearch/pytorchvideo'
    },
}

# Recommended models for different use cases
RECOMMENDED = {
    'lightweight': ['x3d_xs', 'x3d_s'],
    'balanced': ['s3d_kinetics400', 'x3d_m', 'i3d_rgb_kinetics400'],
    'accurate': ['slowfast_r50', 'video_swin_base_k400', 'mvit_base_16x4'],
    'transformer': ['video_swin_tiny_k400', 'video_swin_base_k400', 'mvit_base_16x4'],
    'default': ['s3d_kinetics400', 'i3d_rgb_kinetics400', 'x3d_m', 'video_swin_tiny_k400'],
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
                size_mb = count * block_size / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {percent}% ({size_mb:.1f}/{total_mb:.1f} MB)")
                sys.stdout.flush()
            else:
                size_mb = count * block_size / (1024 * 1024)
                sys.stdout.write(f"\r  Downloaded: {size_mb:.1f} MB")
                sys.stdout.flush()
        
        # Add headers to avoid 403 errors
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
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
    """Download a specific video model."""
    if model_key not in VIDEO_MODEL_WEIGHTS:
        print(f"Error: Unknown model '{model_key}'")
        print(f"Available models: {', '.join(VIDEO_MODEL_WEIGHTS.keys())}")
        return False
    
    model_info = VIDEO_MODEL_WEIGHTS[model_key]
    output_path = Path(output_dir) / model_info['filename']
    
    # Skip if already exists
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ {model_info['filename']} already exists ({size_mb:.1f} MB), skipping...")
        return True
    
    return download_file(
        url=model_info['url'],
        destination=str(output_path),
        description=model_info['description']
    )


def list_models():
    """List all available models."""
    print("\n" + "=" * 70)
    print("Available Video Model Weights")
    print("=" * 70)
    
    # Group by model type
    model_groups = {
        'S3D': [],
        'I3D': [],
        'X3D': [],
        'SlowFast': [],
        'Slow': [],
        'C2D': [],
        'CSN': [],
        'R(2+1)D': [],
        'Video Swin': [],
        'MViT': [],
    }
    
    for key, info in VIDEO_MODEL_WEIGHTS.items():
        if key.startswith('s3d'):
            model_groups['S3D'].append((key, info))
        elif key.startswith('i3d'):
            model_groups['I3D'].append((key, info))
        elif key.startswith('x3d'):
            model_groups['X3D'].append((key, info))
        elif key.startswith('slowfast'):
            model_groups['SlowFast'].append((key, info))
        elif key.startswith('slow_'):
            model_groups['Slow'].append((key, info))
        elif key.startswith('c2d'):
            model_groups['C2D'].append((key, info))
        elif key.startswith('csn'):
            model_groups['CSN'].append((key, info))
        elif key.startswith('r2plus1d'):
            model_groups['R(2+1)D'].append((key, info))
        elif key.startswith('video_swin'):
            model_groups['Video Swin'].append((key, info))
        elif key.startswith('mvit'):
            model_groups['MViT'].append((key, info))
    
    for group_name, models in model_groups.items():
        if models:
            print(f"\n{group_name}:")
            print("-" * 60)
            for key, info in models:
                print(f"  {key:30} - {info['description']}")
    
    print("\n" + "=" * 70)
    print("Recommended sets:")
    print("-" * 70)
    for use_case, models in RECOMMENDED.items():
        print(f"  {use_case:15} -> {', '.join(models)}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Download pretrained video model weights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download S3D model
  python download_video_model_weights.py --model s3d_kinetics400

  # Download multiple models
  python download_video_model_weights.py --model s3d_kinetics400 --model i3d_rgb_kinetics400 --model x3d_m

  # Download default recommended models (S3D, I3D, X3D-M)
  python download_video_model_weights.py --recommended default

  # Download transformer models
  python download_video_model_weights.py --recommended transformer

  # Download all available models
  python download_video_model_weights.py --all

  # List available models
  python download_video_model_weights.py --list
        """
    )
    
    parser.add_argument('--model', type=str, action='append', default=[],
                        help='Model(s) to download (can specify multiple)')
    parser.add_argument('--all', action='store_true',
                        help='Download all available models')
    parser.add_argument('--recommended', type=str, 
                        choices=['lightweight', 'balanced', 'accurate', 'transformer', 'default'],
                        help='Download recommended model set')
    parser.add_argument('--output_dir', type=str, default='./pretrained_models',
                        help='Output directory for weights (default: ./pretrained_models)')
    parser.add_argument('--list', action='store_true',
                        help='List available models and exit')
    
    args = parser.parse_args()
    
    # List models
    if args.list:
        list_models()
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which models to download
    models_to_download = []
    
    if args.all:
        models_to_download = list(VIDEO_MODEL_WEIGHTS.keys())
    elif args.recommended:
        models_to_download = RECOMMENDED[args.recommended]
    elif args.model:
        models_to_download = args.model
    else:
        # Default: download common models
        models_to_download = RECOMMENDED['default']
        print("No model specified. Downloading default models: S3D, I3D-RGB, X3D-M")
        print("Use --help to see all options, or --list to see available models.")
    
    print("\n" + "=" * 70)
    print("Video Model Pretrained Weights Downloader")
    print("=" * 70)
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
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Output directory: {output_dir.absolute()}")
    
    # List downloaded files
    print("\nDownloaded files:")
    for f in sorted(output_dir.glob('*.pt*')):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")
    
    print("=" * 70 + "\n")
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
