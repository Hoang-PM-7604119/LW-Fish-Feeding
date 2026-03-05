#!/usr/bin/env python3
"""
Comprehensive Video Model Training Script (Single Modality).

Trains available video model variants with:
- Fixed train/val/test splits
- 50 epochs per model
- WandB logging (one run per model)
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


_PRETRAINED_BASE = "/mnt/disk1/backup_user/hoang.pm/pretrained_models/video"

VIDEO_MODELS = {
    "s3d": {
        "type": "s3d",
        "output_dim": 1024,
        "description": "S3D - Spatiotemporal 3D CNN",
        "pretrained_path": f"{_PRETRAINED_BASE}/S3D_kinetics400.pth",
    },
    "x3d_xs": {
        "type": "x3d",
        "output_dim": 2048,
        "description": "X3D-XS - Efficient 3D CNN",
        "kwargs": {"variant": "xs"},
        "pretrained_path": f"{_PRETRAINED_BASE}/X3D_XS_kinetics400.pyth",
    },
    "x3d_s": {
        "type": "x3d",
        "output_dim": 2048,
        "description": "X3D-S - Efficient 3D CNN",
        "kwargs": {"variant": "s"},
        "pretrained_path": f"{_PRETRAINED_BASE}/X3D_S_kinetics400.pyth",
    },
    "x3d_m": {
        "type": "x3d",
        "output_dim": 2048,
        "description": "X3D-M - Efficient 3D CNN",
        "kwargs": {"variant": "m"},
        "pretrained_path": f"{_PRETRAINED_BASE}/X3D_M_kinetics400.pyth",
    },
    "x3d_l": {
        "type": "x3d",
        "output_dim": 2048,
        "description": "X3D-L - Efficient 3D CNN",
        "kwargs": {"variant": "l"},
        "pretrained_path": f"{_PRETRAINED_BASE}/X3D_L_kinetics400.pyth",
    },
    "movinet_a0": {
        "type": "movinet",
        "output_dim": 2048,
        "description": "MoViNet-A0 - Mobile video model",
        "kwargs": {"variant": "a0"},
        "pretrained_path": f"{_PRETRAINED_BASE}/movinet_a0_base.pt",
    },
    "movinet_a1": {
        "type": "movinet",
        "output_dim": 2048,
        "description": "MoViNet-A1 - Mobile video model",
        "kwargs": {"variant": "a1"},
        "pretrained_path": f"{_PRETRAINED_BASE}/movinet_a1_base.pt",
    },
    "movinet_a2": {
        "type": "movinet",
        "output_dim": 2048,
        "description": "MoViNet-A2 - Mobile video model",
        "kwargs": {"variant": "a2"},
        "pretrained_path": f"{_PRETRAINED_BASE}/movinet_a2_base.pt",
    },
    "movinet_a3": {
        "type": "movinet",
        "output_dim": 2048,
        "description": "MoViNet-A3 - Mobile video model",
        "kwargs": {"variant": "a3"},
        "pretrained_path": f"{_PRETRAINED_BASE}/movinet_a3_base.pt",
    },
    "movinet_a4": {
        "type": "movinet",
        "output_dim": 2048,
        "description": "MoViNet-A4 - Mobile video model",
        "kwargs": {"variant": "a4"},
        "pretrained_path": f"{_PRETRAINED_BASE}/movinet_a4_base.pt",
    },
    "movinet_a5": {
        "type": "movinet",
        "output_dim": 2048,
        "description": "MoViNet-A5 - Mobile video model",
        "kwargs": {"variant": "a5"},
        "pretrained_path": f"{_PRETRAINED_BASE}/movinet_a5_base.pt",
    },
    "i3d": {
        "type": "i3d",
        "output_dim": 1024,
        "description": "I3D - Inflated 3D ConvNet (Kinetics)",
        "kwargs": {},
        "pretrained_path": f"{_PRETRAINED_BASE}/I3D_8x8_R50_kinetics400.pyth",
    },
    "videomae": {
        "type": "videomae",
        "output_dim": 768,
        "description": "VideoMAE - Masked Autoencoder (placeholder)",
        "kwargs": {},
    },
}


def generate_config(model_name, model_config, base_config):
    config = {
        "model": {
            "modality": "video",
            "video_encoder": {
                "type": model_config["type"],
                "output_dim": model_config["output_dim"],
                "kwargs": model_config.get("kwargs", {}),
            },
            "classifier": {
                "num_classes": 4,
                "dropout": base_config["dropout"],
            },
        },
        "data": {
            "video_dir": base_config["video_dir"],
            "audio_dir": base_config["audio_dir"],
            "split_file": base_config["split_file"],
            "seed": base_config["seed"],
            "test_sample_per_class": 700,
            "batch_size": base_config["batch_size"],
            "num_workers": base_config["num_workers"],
            "audio_duration": base_config["audio_duration"],
            "sample_rate": base_config["sample_rate"],
            "class_names": ["none", "weak", "medium", "strong"],
        },
        "training": {
            "epochs": base_config["epochs"],
            "learning_rate": base_config["learning_rate"],
            "weight_decay": base_config["weight_decay"],
            "optimizer": "adam",
            "scheduler": "cosine",
            "warmup_epochs": 5,
            "gradient_clip": 1.0,
            "mixed_precision": True,
            "checkpoint_dir": os.path.join(
                base_config["output_dir"], "checkpoints", f"video_{model_name}"
            ),
            "early_stopping": {
                "enabled": False,
                "patience": 15,
                "min_delta": 0.001,
            },
        },
        "logging": {
            "use_wandb": base_config["use_wandb"],
            "wandb_project": base_config["wandb_project"],
            "wandb_entity": base_config.get("wandb_entity"),
            "log_every": 10,
        },
        "evaluation": {
            "test_batch_size": 64,
            "save_predictions": True,
            "save_confusion_matrix": True,
        },
        "hardware": {
            "device": "cuda",
            "gpu_id": 0,
            "deterministic": True,
            "seed": base_config["seed"],
        },
    }

    if "pretrained_path" in model_config:
        config["model"]["video_encoder"]["pretrained_path"] = model_config["pretrained_path"]

    return config


def train_model(config_path, model_name):
    print(f"\n{'='*80}")
    print(f"Training: {model_name.upper()}")
    print(f"Config: {config_path}")
    print(f"{'='*80}\n")

    script_path = Path(__file__).parent / "train_single.py"
    cmd = [sys.executable, str(script_path), "--config", config_path]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()
    return process.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Video Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--video_dir",
        type=str,
        default="/mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_video_uniform",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="/mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/processed_audio",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="/mnt/disk1/backup_user/hoang.pm/UFFIA_data/fixed/splits_uniform/splits.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/disk1/backup_user/hoang.pm/UFFIA_data/experiments/video_single_uniform",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=list(VIDEO_MODELS.keys()) + ["all"],
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio_duration", type=float, default=2.0)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--wandb_project", type=str, default="Multimodalities for Aquaculture"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="hoangpmh2406-vinuniversity"
    )

    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    if args.no_wandb:
        args.use_wandb = False

    # Decide which models to train: all video models (S3D, X3D, I3D, MoViNet, VideoMAE, ...)
    if args.models is None or "all" in args.models:
        models_to_train = list(VIDEO_MODELS.keys())
    else:
        models_to_train = args.models

    # X3D/I3D use pytorchvideo; MoViNet uses movinets package. No skip based on torchvision.

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "configs").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    base_config = {
        "video_dir": args.video_dir,
        "audio_dir": args.audio_dir,
        "split_file": args.split_file,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "audio_duration": args.audio_duration,
        "sample_rate": args.sample_rate,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
    }

    print("\n" + "=" * 80)
    print("COMPREHENSIVE VIDEO MODEL TRAINING")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split file: {args.split_file}")
    print(f"\nModels to train ({len(models_to_train)}):")
    for model in models_to_train:
        print(f"  - {model}: {VIDEO_MODELS[model]['description']}")
    print(f"\nTraining settings:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  WandB: {'Enabled' if args.use_wandb else 'Disabled'}")
    if args.use_wandb:
        print(f"  WandB project: {args.wandb_project}")
        print(f"  WandB entity: {args.wandb_entity}")

    config_paths = {}
    for model_name in models_to_train:
        model_config = VIDEO_MODELS[model_name]
        config = generate_config(model_name, model_config, base_config)

        config_path = output_dir / "configs" / f"video_{model_name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        config_paths[model_name] = str(config_path)
        print(f"✓ Generated: {config_path}")

    experiment_info = {
        "timestamp": datetime.now().isoformat(),
        "models": models_to_train,
        "base_config": base_config,
        "config_files": config_paths,
        "split_file": args.split_file,
    }

    with open(output_dir / "experiment_info.json", "w") as f:
        json.dump(experiment_info, f, indent=2)

    if args.dry_run:
        print("\nDRY RUN - Configs generated, skipping training")
        print("To train manually, run:")
        for model_name, config_path in config_paths.items():
            print(f"  python scripts/train_single.py --config {config_path}")
        return

    results = {}
    for i, model_name in enumerate(models_to_train, 1):
        print(f"\n[{i}/{len(models_to_train)}] Training {model_name}...")
        try:
            return_code = train_model(config_paths[model_name], model_name)
            if return_code == 0:
                results_path = (
                    output_dir / "checkpoints" / f"video_{model_name}" / "test_results.json"
                )
                if results_path.exists():
                    with open(results_path, "r") as f:
                        test_results = json.load(f)
                    results[model_name] = {
                        "status": "success",
                        "test_accuracy": test_results.get("test_accuracy"),
                        "test_f1": test_results.get("test_f1"),
                    }
                    print(
                        f"✓ {model_name}: Accuracy={test_results.get('test_accuracy', 0):.4f}, "
                        f"F1={test_results.get('test_f1', 0):.4f}"
                    )
                else:
                    results[model_name] = {"status": "success", "note": "No results file found"}
            else:
                results[model_name] = {"status": "failed", "return_code": return_code}
                print(f"✗ {model_name}: Training failed with return code {return_code}")
        except Exception as e:
            results[model_name] = {"status": "error", "error": str(e)}
            print(f"✗ {model_name}: Error - {e}")

    final_results = {
        "timestamp": datetime.now().isoformat(),
        "models_trained": len(models_to_train),
        "results": results,
    }

    with open(output_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_dir / 'final_results.json'}")


if __name__ == "__main__":
    main()
