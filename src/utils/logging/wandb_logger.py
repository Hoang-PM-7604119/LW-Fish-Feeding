"""
Weights & Biases (WandB) logger for experiment tracking.

Provides:
- Experiment initialization and configuration logging
- Real-time metrics logging (loss, accuracy, etc.)
- Model architecture logging
- Hyperparameter tracking
- Artifact management
"""

import os
from typing import Dict, Optional, Any
import torch
import torch.nn as nn

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


class WandBLogger:
    """
    Weights & Biases logger for experiment tracking.
    
    Args:
        project: WandB project name
        name: Experiment name
        config: Configuration dictionary
        entity: WandB entity (username or team)
        tags: List of tags for the experiment
        notes: Notes about the experiment
        mode: WandB mode ('online', 'offline', 'disabled')
    """
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        entity: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        mode: str = 'online'
    ):
        self.enabled = WANDB_AVAILABLE and mode != 'disabled'
        
        if not self.enabled:
            print("WandB logging disabled")
            return
        
        # Initialize wandb
        wandb.init(
            project=project,
            name=name,
            config=config,
            entity=entity,
            tags=tags,
            notes=notes,
            mode=mode
        )
        
        print(f"✓ WandB initialized: {wandb.run.get_url()}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to WandB.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number (epoch, iteration)
        """
        if not self.enabled:
            return
        
        wandb.log(metrics, step=step)
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log configuration parameters.
        
        Args:
            config: Configuration dictionary
        """
        if not self.enabled:
            return
        
        wandb.config.update(config)
    
    def log_model_architecture(self, model: nn.Module, input_shapes: Optional[Dict] = None):
        """
        Log model architecture.
        
        Args:
            model: PyTorch model
            input_shapes: Dictionary with input shapes for visualization
        """
        if not self.enabled:
            return
        
        # Log model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.config.update({
            'model_total_params': total_params,
            'model_trainable_params': trainable_params,
            'model_total_params_millions': total_params / 1e6,
        })
        
        # Log model graph if possible
        if input_shapes is not None:
            try:
                # Create dummy inputs
                dummy_inputs = []
                for shape in input_shapes.values():
                    dummy_inputs.append(torch.randn(*shape))
                
                # Log model graph
                wandb.watch(model, log='all', log_freq=100)
            except Exception as e:
                print(f"Warning: Could not log model graph: {e}")
    
    def log_confusion_matrix(
        self,
        y_true: list,
        y_pred: list,
        class_names: list,
        step: Optional[int] = None
    ):
        """
        Log confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            step: Optional step number
        """
        if not self.enabled:
            return
        
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )
        }, step=step)
    
    def log_attention_weights(
        self,
        attention_weights: torch.Tensor,
        name: str = "attention",
        step: Optional[int] = None
    ):
        """
        Log attention weight visualizations.
        
        Args:
            attention_weights: Attention weights tensor
            name: Name for the visualization
            step: Optional step number
        """
        if not self.enabled:
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Convert to numpy
            attn = attention_weights.detach().cpu().numpy()
            
            # Plot attention map
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(attn[0, 0], cmap='viridis', ax=ax)
            ax.set_title(f"Attention Weights: {name}")
            
            wandb.log({f"attention/{name}": wandb.Image(fig)}, step=step)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not log attention weights: {e}")
    
    def log_checkpoint(
        self,
        checkpoint_path: str,
        metadata: Optional[Dict] = None,
        is_best: bool = False
    ):
        """
        Log model checkpoint as artifact.
        
        Args:
            checkpoint_path: Path to checkpoint file
            metadata: Optional metadata dictionary
            is_best: Whether this is the best checkpoint
        """
        if not self.enabled:
            return
        
        artifact_name = "model-best" if is_best else "model-checkpoint"
        artifact = wandb.Artifact(artifact_name, type='model', metadata=metadata)
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
    
    def log_table(
        self,
        name: str,
        columns: list,
        data: list,
        step: Optional[int] = None
    ):
        """
        Log data as table.
        
        Args:
            name: Table name
            columns: List of column names
            data: List of rows
            step: Optional step number
        """
        if not self.enabled:
            return
        
        table = wandb.Table(columns=columns, data=data)
        wandb.log({name: table}, step=step)
    
    def finish(self):
        """Finish WandB run."""
        if not self.enabled:
            return
        
        wandb.finish()
        print("✓ WandB run finished")


def init_wandb(
    project: str,
    name: Optional[str] = None,
    config: Optional[Dict] = None,
    **kwargs
) -> WandBLogger:
    """
    Initialize WandB logger (convenience function).
    
    Args:
        project: Project name
        name: Experiment name
        config: Configuration dictionary
        **kwargs: Additional arguments for WandBLogger
        
    Returns:
        logger: WandBLogger instance
    """
    return WandBLogger(project=project, name=name, config=config, **kwargs)


if __name__ == '__main__':
    """Test WandB logger."""
    print("="*80)
    print("Testing WandB Logger")
    print("="*80)
    
    # Initialize logger (offline mode for testing)
    logger = WandBLogger(
        project="test_project",
        name="test_experiment",
        config={'learning_rate': 0.001, 'batch_size': 32},
        mode='disabled'  # Use 'disabled' to avoid actual logging
    )
    
    # Log some metrics
    for epoch in range(5):
        metrics = {
            'train_loss': 1.0 - epoch * 0.1,
            'train_acc': 0.5 + epoch * 0.08,
            'val_loss': 1.1 - epoch * 0.09,
            'val_acc': 0.45 + epoch * 0.09,
        }
        logger.log_metrics(metrics, step=epoch)
        print(f"Epoch {epoch}: {metrics}")
    
    # Finish
    logger.finish()
    
    print("\n" + "="*80)
    print("✓ WandB logger test completed!")
    print("="*80)
