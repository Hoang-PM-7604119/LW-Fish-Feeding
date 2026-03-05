#!/usr/bin/env python3
"""
Comprehensive Pretrained Model Downloader for LW_fish Project

This script downloads all pretrained models used in the project:
- Video models: S3D, I3D, X3D, SlowFast, Video Swin, MViT, MoViNet
- Audio models: PANNs (CNN6, CNN10, CNN14, ResNet, MobileNet), AST

Usage:
    python download_pretrained_models.py --all              # Download all models
    python download_pretrained_models.py --video            # Download video models only
    python download_pretrained_models.py --audio            # Download audio models only
    python download_pretrained_models.py --model s3d i3d    # Download specific models
    python download_pretrained_models.py --list             # List all available models

Default output directory: /mnt/disk1/backup_user/hoang.pm/pretrained_models
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse
import requests
from tqdm import tqdm


# ============================================================================
# Model Registry - All pretrained models with download URLs
# ============================================================================

VIDEO_MODELS: Dict[str, Dict] = {
    # S3D
    "s3d": {
        "filename": "S3D_kinetics400.pth",
        "url": "https://download.pytorch.org/models/s3d-d76dad2f.pth",
        "category": "video",
        "description": "S3D pretrained on Kinetics-400",
    },
    
    # I3D Models
    "i3d_rgb_imagenet": {
        "filename": "I3D_rgb_imagenet_kinetics.pt",
        "url": "https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt",
        "category": "video",
        "description": "I3D RGB pretrained on ImageNet + Kinetics",
    },
    "i3d_flow_imagenet": {
        "filename": "I3D_flow_imagenet_kinetics.pt",
        "url": "https://github.com/piergiaj/pytorch-i3d/raw/master/models/flow_imagenet.pt",
        "category": "video",
        "description": "I3D Flow pretrained on ImageNet + Kinetics",
    },
    "i3d_r50": {
        "filename": "I3D_8x8_R50_kinetics400.pyth",
        "url": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/I3D_8x8_R50.pyth",
        "category": "video",
        "description": "I3D ResNet50 pretrained on Kinetics-400",
    },
    
    # X3D Models
    "x3d_xs": {
        "filename": "X3D_XS_kinetics400.pyth",
        "url": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_XS.pyth",
        "category": "video",
        "description": "X3D-XS pretrained on Kinetics-400",
    },
    "x3d_s": {
        "filename": "X3D_S_kinetics400.pyth",
        "url": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_S.pyth",
        "category": "video",
        "description": "X3D-S pretrained on Kinetics-400",
    },
    "x3d_m": {
        "filename": "X3D_M_kinetics400.pyth",
        "url": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_M.pyth",
        "category": "video",
        "description": "X3D-M pretrained on Kinetics-400",
    },
    "x3d_l": {
        "filename": "X3D_L_kinetics400.pyth",
        "url": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_L.pyth",
        "category": "video",
        "description": "X3D-L pretrained on Kinetics-400",
    },
    
    # SlowFast Models
    "slowfast_r50": {
        "filename": "SlowFast_8x8_R50_kinetics400.pyth",
        "url": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST_8x8_R50.pyth",
        "category": "video",
        "description": "SlowFast ResNet50 pretrained on Kinetics-400",
    },
    "slowfast_r101": {
        "filename": "SlowFast_16x8_R101_kinetics400.pyth",
        "url": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST_16x8_R101_50_50.pyth",
        "category": "video",
        "description": "SlowFast ResNet101 pretrained on Kinetics-400",
    },
    "slow_r50": {
        "filename": "Slow_8x8_R50_kinetics400.pyth",
        "url": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW_8x8_R50.pyth",
        "category": "video",
        "description": "Slow pathway ResNet50 pretrained on Kinetics-400",
    },
    
    # C2D Model
    "c2d_r50": {
        "filename": "C2D_8x8_R50_kinetics400.pyth",
        "url": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/C2D_8x8_R50.pyth",
        "category": "video",
        "description": "C2D ResNet50 pretrained on Kinetics-400",
    },
    
    # CSN Model
    "csn_r101": {
        "filename": "CSN_32x2_R101_kinetics400.pyth",
        "url": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/CSN_32x2_R101.pyth",
        "category": "video",
        "description": "CSN ResNet101 pretrained on Kinetics-400",
    },
    
    # R(2+1)D Model
    "r2plus1d_r50": {
        "filename": "R2plus1D_16x4_R50_kinetics400.pyth",
        "url": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/R2PLUS1D_16x4_R50.pyth",
        "category": "video",
        "description": "R(2+1)D ResNet50 pretrained on Kinetics-400",
    },
    
    # Video Swin Transformer Models
    "video_swin_tiny": {
        "filename": "VideoSwin_tiny_k400.pth",
        "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth",
        "category": "video",
        "description": "Video Swin Tiny pretrained on Kinetics-400",
    },
    "video_swin_small": {
        "filename": "VideoSwin_small_k400.pth",
        "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_small_patch244_window877_kinetics400_1k.pth",
        "category": "video",
        "description": "Video Swin Small pretrained on Kinetics-400",
    },
    "video_swin_base_k400": {
        "filename": "VideoSwin_base_k400.pth",
        "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_1k.pth",
        "category": "video",
        "description": "Video Swin Base pretrained on Kinetics-400",
    },
    "video_swin_base_k600": {
        "filename": "VideoSwin_base_k600.pth",
        "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics600_22k.pth",
        "category": "video",
        "description": "Video Swin Base pretrained on Kinetics-600",
    },
    
    # MViT Models
    "mvit_b_16x4": {
        "filename": "MViT_B_16x4_kinetics400.pyth",
        "url": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/MVIT_B_16x4.pyth",
        "category": "video",
        "description": "MViT-B 16x4 pretrained on Kinetics-400",
    },
    "mvit_b_32x3": {
        "filename": "MViT_B_32x3_kinetics400.pyth",
        "url": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/MVIT_B_32x3.pyth",
        "category": "video",
        "description": "MViT-B 32x3 pretrained on Kinetics-400",
    },
    
    # MoViNet Models (Base variants)
    "movinet_a0_base": {
        "filename": "movinet_a0_base.pt",
        "url": "https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA0_statedict_v3",
        "category": "video",
        "description": "MoViNet-A0 Base pretrained on Kinetics-600",
    },
    "movinet_a1_base": {
        "filename": "movinet_a1_base.pt",
        "url": "https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA1_statedict_v3",
        "category": "video",
        "description": "MoViNet-A1 Base pretrained on Kinetics-600",
    },
    "movinet_a2_base": {
        "filename": "movinet_a2_base.pt",
        "url": "https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA2_statedict_v3",
        "category": "video",
        "description": "MoViNet-A2 Base pretrained on Kinetics-600",
    },
    "movinet_a3_base": {
        "filename": "movinet_a3_base.pt",
        "url": "https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA3_statedict_v3",
        "category": "video",
        "description": "MoViNet-A3 Base pretrained on Kinetics-600",
    },
    "movinet_a4_base": {
        "filename": "movinet_a4_base.pt",
        "url": "https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA4_statedict_v3",
        "category": "video",
        "description": "MoViNet-A4 Base pretrained on Kinetics-600",
    },
    "movinet_a5_base": {
        "filename": "movinet_a5_base.pt",
        "url": "https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA5_statedict_v3",
        "category": "video",
        "description": "MoViNet-A5 Base pretrained on Kinetics-600",
    },
    
    # MoViNet Models (Stream variants)
    "movinet_a0_stream": {
        "filename": "movinet_a0_stream.pt",
        "url": "https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA0_stream_statedict_v3",
        "category": "video",
        "description": "MoViNet-A0 Stream pretrained on Kinetics-600",
    },
    "movinet_a1_stream": {
        "filename": "movinet_a1_stream.pt",
        "url": "https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA1_stream_statedict_v3",
        "category": "video",
        "description": "MoViNet-A1 Stream pretrained on Kinetics-600",
    },
    "movinet_a2_stream": {
        "filename": "movinet_a2_stream.pt",
        "url": "https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA2_stream_statedict_v3",
        "category": "video",
        "description": "MoViNet-A2 Stream pretrained on Kinetics-600",
    },
}

AUDIO_MODELS: Dict[str, Dict] = {
    # PANNs Models (Pretrained Audio Neural Networks)
    "panns_cnn6": {
        "filename": "Cnn6_mAP=0.343.pth",
        "url": "https://zenodo.org/record/3987831/files/Cnn6_mAP%3D0.343.pth",
        "category": "audio",
        "description": "PANNs CNN6 pretrained on AudioSet",
    },
    "panns_cnn10": {
        "filename": "Cnn10_mAP=0.380.pth",
        "url": "https://zenodo.org/record/3987831/files/Cnn10_mAP%3D0.380.pth",
        "category": "audio",
        "description": "PANNs CNN10 pretrained on AudioSet",
    },
    "panns_cnn14": {
        "filename": "Cnn14_mAP=0.431.pth",
        "url": "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth",
        "category": "audio",
        "description": "PANNs CNN14 pretrained on AudioSet",
    },
    "panns_cnn14_16k": {
        "filename": "Cnn14_16k_mAP=0.438.pth",
        "url": "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth",
        "category": "audio",
        "description": "PANNs CNN14 (16kHz) pretrained on AudioSet",
    },
    "panns_cnn14_decision": {
        "filename": "Cnn14_DecisionLevelMax_mAP=0.385.pth",
        "url": "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth",
        "category": "audio",
        "description": "PANNs CNN14 Decision Level Max pretrained on AudioSet",
    },
    "panns_resnet22": {
        "filename": "ResNet22_mAP=0.430.pth",
        "url": "https://zenodo.org/record/3987831/files/ResNet22_mAP%3D0.430.pth",
        "category": "audio",
        "description": "PANNs ResNet22 pretrained on AudioSet",
    },
    "panns_resnet38": {
        "filename": "ResNet38_mAP=0.434.pth",
        "url": "https://zenodo.org/record/3987831/files/ResNet38_mAP%3D0.434.pth",
        "category": "audio",
        "description": "PANNs ResNet38 pretrained on AudioSet",
    },
    "panns_resnet54": {
        "filename": "ResNet54_mAP=0.429.pth",
        "url": "https://zenodo.org/record/3987831/files/ResNet54_mAP%3D0.429.pth",
        "category": "audio",
        "description": "PANNs ResNet54 pretrained on AudioSet",
    },
    "panns_mobilenetv1": {
        "filename": "MobileNetV1_mAP=0.389.pth",
        "url": "https://zenodo.org/record/3987831/files/MobileNetV1_mAP%3D0.389.pth",
        "category": "audio",
        "description": "PANNs MobileNetV1 pretrained on AudioSet",
    },
    "panns_mobilenetv2": {
        "filename": "MobileNetV2_mAP=0.383.pth",
        "url": "https://zenodo.org/record/3987831/files/MobileNetV2_mAP%3D0.383.pth",
        "category": "audio",
        "description": "PANNs MobileNetV2 pretrained on AudioSet",
    },
    
    # AST (Audio Spectrogram Transformer) Models
    "ast_audioset_10_10_0.4495": {
        "filename": "audioset_10_10_0.4495.pth",
        "url": "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4495.pth?dl=1",
        "category": "audio",
        "description": "AST pretrained on AudioSet (mAP=0.4495)",
    },
    "ast_audioset_10_10_0.4593": {
        "filename": "audioset_10_10_0.4593.pth",
        "url": "https://www.dropbox.com/s/1tv0hovue1bxupk/audioset_0.4593.pth?dl=1",
        "category": "audio",
        "description": "AST pretrained on AudioSet (mAP=0.4593)",
    },
    "ast_speech_commands": {
        "filename": "speech_commands_v2_35_10_0.9812.pth",
        "url": "https://www.dropbox.com/s/q0tbqpwv44pquwy/speechcommandv2_35_10.pth?dl=1",
        "category": "audio",
        "description": "AST pretrained on Speech Commands V2",
    },
}

# Combine all models
ALL_MODELS = {**VIDEO_MODELS, **AUDIO_MODELS}


# ============================================================================
# Download Functions
# ============================================================================

def get_file_size(url: str) -> Optional[int]:
    """Get file size from URL headers without downloading."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        if response.status_code == 200:
            return int(response.headers.get('content-length', 0))
    except Exception:
        pass
    return None


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def download_file(url: str, dest_path: Path, desc: str = None, force: bool = False) -> bool:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: Download URL
        dest_path: Destination file path
        desc: Description for progress bar
        force: Force re-download even if file exists
        
    Returns:
        True if download successful, False otherwise
    """
    if dest_path.exists() and not force:
        print(f"  [SKIP] {dest_path.name} already exists")
        return True
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        desc = desc or dest_path.name
        
        with open(dest_path, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=f"  {desc}",
                ncols=80
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"  [OK] Downloaded {dest_path.name}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] Failed to download {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def download_model(
    model_key: str,
    output_dir: Path,
    force: bool = False
) -> bool:
    """
    Download a specific model by key.
    
    Args:
        model_key: Model identifier from registry
        output_dir: Base output directory
        force: Force re-download
        
    Returns:
        True if successful, False otherwise
    """
    if model_key not in ALL_MODELS:
        print(f"[ERROR] Unknown model: {model_key}")
        return False
    
    model_info = ALL_MODELS[model_key]
    category = model_info["category"]
    filename = model_info["filename"]
    url = model_info["url"]
    
    dest_path = output_dir / category / filename
    
    return download_file(url, dest_path, filename, force)


def download_category(
    category: str,
    output_dir: Path,
    force: bool = False
) -> Dict[str, bool]:
    """
    Download all models in a category.
    
    Args:
        category: 'video' or 'audio'
        output_dir: Base output directory
        force: Force re-download
        
    Returns:
        Dict mapping model_key to success status
    """
    models = VIDEO_MODELS if category == "video" else AUDIO_MODELS
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Downloading {category.upper()} models")
    print(f"{'='*60}")
    
    for model_key, model_info in models.items():
        print(f"\n[{model_key}] {model_info['description']}")
        results[model_key] = download_model(model_key, output_dir, force)
    
    return results


def list_models():
    """Print all available models."""
    print("\n" + "="*80)
    print("AVAILABLE PRETRAINED MODELS")
    print("="*80)
    
    print("\n" + "-"*40)
    print("VIDEO MODELS")
    print("-"*40)
    for key, info in VIDEO_MODELS.items():
        print(f"  {key:25s} - {info['description']}")
    
    print("\n" + "-"*40)
    print("AUDIO MODELS")
    print("-"*40)
    for key, info in AUDIO_MODELS.items():
        print(f"  {key:25s} - {info['description']}")
    
    print("\n" + "="*80)
    print(f"Total: {len(VIDEO_MODELS)} video models, {len(AUDIO_MODELS)} audio models")
    print("="*80)


def print_summary(results: Dict[str, bool]):
    """Print download summary."""
    successful = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"  Successful: {successful}")
    print(f"  Failed:     {failed}")
    print(f"  Total:      {len(results)}")
    
    if failed > 0:
        print("\nFailed downloads:")
        for model_key, success in results.items():
            if not success:
                print(f"  - {model_key}")


# ============================================================================
# Model Sets for Common Use Cases
# ============================================================================

MODEL_SETS = {
    "essential": [
        "s3d", "i3d_rgb_imagenet", "x3d_m",
        "panns_cnn10", "panns_cnn14"
    ],
    "lightweight": [
        "movinet_a0_base", "movinet_a2_base",
        "panns_cnn6", "panns_mobilenetv2"
    ],
    "transformers": [
        "video_swin_tiny", "video_swin_base_k400",
        "mvit_b_16x4", "ast_audioset_10_10_0.4593"
    ],
    "panns_all": [
        "panns_cnn6", "panns_cnn10", "panns_cnn14",
        "panns_cnn14_16k", "panns_cnn14_decision",
        "panns_resnet22", "panns_resnet38", "panns_resnet54",
        "panns_mobilenetv1", "panns_mobilenetv2"
    ],
    "movinet_all": [
        "movinet_a0_base", "movinet_a1_base", "movinet_a2_base",
        "movinet_a3_base", "movinet_a4_base", "movinet_a5_base",
        "movinet_a0_stream", "movinet_a1_stream", "movinet_a2_stream"
    ],
    "x3d_all": ["x3d_xs", "x3d_s", "x3d_m", "x3d_l"],
}


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained models for LW_fish project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                     Download all models
  %(prog)s --video                   Download video models only
  %(prog)s --audio                   Download audio models only
  %(prog)s --model s3d panns_cnn10   Download specific models
  %(prog)s --set essential           Download essential model set
  %(prog)s --list                    List all available models
  %(prog)s --sets                    List all model sets
        """
    )
    
    parser.add_argument(
        "--all", action="store_true",
        help="Download all models"
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Download all video models"
    )
    parser.add_argument(
        "--audio", action="store_true",
        help="Download all audio models"
    )
    parser.add_argument(
        "--model", nargs="+", metavar="NAME",
        help="Download specific models by name"
    )
    parser.add_argument(
        "--set", choices=list(MODEL_SETS.keys()),
        help="Download a predefined model set"
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path,
        default=Path("/mnt/disk1/backup_user/hoang.pm/pretrained_models"),
        help="Output directory (default: /mnt/disk1/backup_user/hoang.pm/pretrained_models)"
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force re-download existing files"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available models"
    )
    parser.add_argument(
        "--sets", action="store_true",
        help="List all model sets"
    )
    
    args = parser.parse_args()
    
    # Handle list commands
    if args.list:
        list_models()
        return 0
    
    if args.sets:
        print("\nAvailable Model Sets:")
        print("-" * 40)
        for set_name, models in MODEL_SETS.items():
            print(f"\n  {set_name}:")
            for model in models:
                print(f"    - {model}")
        return 0
    
    # Check if any download option specified
    if not any([args.all, args.video, args.audio, args.model, args.set]):
        parser.print_help()
        return 1
    
    output_dir = args.output_dir.resolve()
    print(f"\nOutput directory: {output_dir}")
    
    results = {}
    
    # Download based on options
    if args.all:
        results.update(download_category("video", output_dir, args.force))
        results.update(download_category("audio", output_dir, args.force))
    
    elif args.video:
        results.update(download_category("video", output_dir, args.force))
    
    elif args.audio:
        results.update(download_category("audio", output_dir, args.force))
    
    elif args.set:
        print(f"\nDownloading model set: {args.set}")
        print("="*60)
        for model_key in MODEL_SETS[args.set]:
            model_info = ALL_MODELS.get(model_key, {})
            print(f"\n[{model_key}] {model_info.get('description', 'Unknown')}")
            results[model_key] = download_model(model_key, output_dir, args.force)
    
    elif args.model:
        print("\nDownloading specified models")
        print("="*60)
        for model_key in args.model:
            model_info = ALL_MODELS.get(model_key, {})
            print(f"\n[{model_key}] {model_info.get('description', 'Unknown')}")
            results[model_key] = download_model(model_key, output_dir, args.force)
    
    # Print summary
    if results:
        print_summary(results)
        return 0 if all(results.values()) else 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
