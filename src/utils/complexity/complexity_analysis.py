"""
Model complexity analysis.

Provides:
- Parameter counting (total, trainable, frozen)
- GFLOPs calculation
- Model size estimation
- Layer-wise analysis
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from thop import profile, clever_format


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        params_dict: Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'total_millions': total_params / 1e6,
        'trainable_millions': trainable_params / 1e6,
    }


def calculate_gflops(
    model: nn.Module,
    video_input: torch.Tensor,
    audio_input: torch.Tensor,
    verbose: bool = False
) -> Tuple[float, float]:
    """
    Calculate GFLOPs (Giga Floating Point Operations).
    
    Args:
        model: PyTorch model
        video_input: Sample video input tensor
        audio_input: Sample audio input tensor
        verbose: Print detailed information
        
    Returns:
        gflops: GFLOPs count
        params_millions: Parameters in millions
    """
    model.eval()
    
    try:
        # Use thop library to profile
        flops, params = profile(
            model,
            inputs=(video_input, audio_input),
            verbose=verbose
        )
        
        gflops = flops / 1e9  # Convert to GFLOPs
        params_millions = params / 1e6
        
        return gflops, params_millions
        
    except Exception as e:
        print(f"Warning: GFLOPs calculation failed: {e}")
        # Fallback to parameter counting only
        params = sum(p.numel() for p in model.parameters())
        return 0.0, params / 1e6


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Estimate model size on disk.
    
    Args:
        model: PyTorch model
        
    Returns:
        size_dict: Dictionary with size estimates in different units
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size_bytes = param_size + buffer_size
    
    return {
        'bytes': total_size_bytes,
        'kb': total_size_bytes / 1024,
        'mb': total_size_bytes / (1024 ** 2),
        'gb': total_size_bytes / (1024 ** 3),
    }


def analyze_model_complexity(
    model: nn.Module,
    video_shape: Tuple[int, ...] = (1, 16, 3, 224, 224),
    audio_shape: Tuple[int, ...] = (1, 64000),
    device: str = 'cpu',
    verbose: bool = True
) -> Dict:
    """
    Comprehensive model complexity analysis.
    
    Args:
        model: PyTorch model
        video_shape: Shape of video input (B, T, C, H, W)
        audio_shape: Shape of audio input (B, n_samples)
        device: Device to run analysis on
        verbose: Print detailed information
        
    Returns:
        analysis: Dictionary with all complexity metrics
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy inputs
    video_input = torch.randn(*video_shape).to(device)
    audio_input = torch.randn(*audio_shape).to(device)
    
    # Count parameters
    params = count_parameters(model)
    
    # Calculate GFLOPs
    gflops, params_millions = calculate_gflops(model, video_input, audio_input, verbose=False)
    
    # Get model size
    size = get_model_size(model)
    
    # Compile results
    analysis = {
        'parameters': params,
        'gflops': gflops,
        'model_size': size,
        'input_shapes': {
            'video': video_shape,
            'audio': audio_shape,
        }
    }
    
    if verbose:
        print_complexity_analysis(analysis)
    
    return analysis


def print_complexity_analysis(analysis: Dict):
    """
    Print complexity analysis in a nicely formatted way.
    
    Args:
        analysis: Dictionary from analyze_model_complexity
    """
    print("\n" + "="*80)
    print("Model Complexity Analysis")
    print("="*80)
    
    # Parameters
    params = analysis['parameters']
    print(f"\nParameters:")
    print(f"  Total:      {params['total']:>15,} ({params['total_millions']:>8.2f}M)")
    print(f"  Trainable:  {params['trainable']:>15,} ({params['trainable_millions']:>8.2f}M)")
    print(f"  Frozen:     {params['frozen']:>15,}")
    
    # GFLOPs
    print(f"\nComputational Complexity:")
    print(f"  GFLOPs:     {analysis['gflops']:>15.2f}")
    
    # Model size
    size = analysis['model_size']
    print(f"\nModel Size:")
    print(f"  MB:         {size['mb']:>15.2f}")
    print(f"  GB:         {size['gb']:>15.4f}")
    
    # Input shapes
    print(f"\nInput Shapes:")
    print(f"  Video:      {analysis['input_shapes']['video']}")
    print(f"  Audio:      {analysis['input_shapes']['audio']}")
    
    print("="*80 + "\n")


def compare_models(models: Dict[str, nn.Module], **kwargs) -> Dict:
    """
    Compare complexity of multiple models.
    
    Args:
        models: Dictionary mapping model names to model instances
        **kwargs: Arguments for analyze_model_complexity
        
    Returns:
        comparison: Dictionary with comparison results
    """
    comparison = {}
    
    print("\n" + "="*80)
    print("Model Comparison")
    print("="*80)
    
    for name, model in models.items():
        print(f"\nAnalyzing {name}...")
        analysis = analyze_model_complexity(model, verbose=False, **kwargs)
        comparison[name] = analysis
    
    # Print comparison table
    print("\n" + "-"*80)
    print(f"{'Model':<30} {'Params (M)':<15} {'GFLOPs':<15} {'Size (MB)':<15}")
    print("-"*80)
    
    for name, analysis in comparison.items():
        params_m = analysis['parameters']['total_millions']
        gflops = analysis['gflops']
        size_mb = analysis['model_size']['mb']
        print(f"{name:<30} {params_m:<15.2f} {gflops:<15.2f} {size_mb:<15.2f}")
    
    print("="*80 + "\n")
    
    return comparison


def layer_wise_analysis(model: nn.Module, max_depth: int = 3) -> Dict:
    """
    Analyze parameter count layer by layer.
    
    Args:
        model: PyTorch model
        max_depth: Maximum depth of module hierarchy to show
        
    Returns:
        layer_info: Dictionary with layer-wise information
    """
    layer_info = {}
    
    def analyze_module(module, name, depth=0):
        if depth > max_depth:
            return
        
        num_params = sum(p.numel() for p in module.parameters(recurse=False))
        if num_params > 0:
            layer_info[name] = {
                'params': num_params,
                'params_millions': num_params / 1e6,
                'trainable': sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad),
            }
        
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            analyze_module(child_module, full_name, depth + 1)
    
    analyze_module(model, "model")
    
    return layer_info


def print_layer_wise_analysis(layer_info: Dict):
    """Print layer-wise analysis."""
    print("\n" + "="*80)
    print("Layer-wise Parameter Analysis")
    print("="*80)
    print(f"{'Layer':<50} {'Parameters':<15} {'Trainable':<15}")
    print("-"*80)
    
    # Sort by number of parameters
    sorted_layers = sorted(layer_info.items(), key=lambda x: x[1]['params'], reverse=True)
    
    for name, info in sorted_layers:
        params = info['params']
        trainable = info['trainable']
        print(f"{name:<50} {params:>14,} {trainable:>14,}")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    """Test complexity analysis."""
    print("="*80)
    print("Testing Complexity Analysis")
    print("="*80)
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.video_encoder = nn.Sequential(
                nn.Conv3d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
            )
            self.audio_encoder = nn.Sequential(
                nn.Linear(64000, 512),
                nn.ReLU(),
            )
            self.fusion = nn.Linear(64 + 512, 256)
            self.classifier = nn.Linear(256, 4)
        
        def forward(self, video, audio):
            v = self.video_encoder(video.permute(0, 2, 1, 3, 4))
            a = self.audio_encoder(audio)
            fused = self.fusion(torch.cat([v, a], dim=-1))
            return self.classifier(fused)
    
    model = SimpleModel()
    
    # Analyze complexity
    analysis = analyze_model_complexity(
        model,
        video_shape=(1, 16, 3, 224, 224),
        audio_shape=(1, 64000),
        verbose=True
    )
    
    # Layer-wise analysis
    layer_info = layer_wise_analysis(model)
    print_layer_wise_analysis(layer_info)
    
    print("\n✓ Complexity analysis test completed!")
