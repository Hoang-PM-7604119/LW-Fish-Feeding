"""
Evaluation metrics for multimodal classification.

Provides:
- Accuracy, Precision, Recall, F1-score
- Per-class metrics
- Confusion matrix
- Average meters for tracking
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    average: str = 'macro'
) -> Dict:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels [N]
        y_pred: Predicted labels [N]
        class_names: List of class names (default: ['none', 'weak', 'medium', 'strong'])
        average: Averaging strategy ('macro', 'micro', 'weighted')
        
    Returns:
        metrics: Dictionary containing all metrics
    """
    if class_names is None:
        class_names = ['none', 'weak', 'medium', 'strong']
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            metrics[f'precision_{class_name}'] = float(precision_per_class[i])
            metrics[f'recall_{class_name}'] = float(recall_per_class[i])
            metrics[f'f1_{class_name}'] = float(f1_per_class[i])
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def calculate_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 4,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate per-class accuracy.
    
    Args:
        y_true: True labels [N]
        y_pred: Predicted labels [N]
        num_classes: Number of classes
        class_names: List of class names
        
    Returns:
        per_class_acc: Dictionary of per-class accuracy
    """
    if class_names is None:
        class_names = ['none', 'weak', 'medium', 'strong']
    
    per_class_acc = {}
    
    for i in range(num_classes):
        mask = (y_true == i)
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).mean()
            per_class_acc[class_names[i]] = float(acc)
        else:
            per_class_acc[class_names[i]] = 0.0
    
    return per_class_acc


def print_metrics(
    metrics: Dict,
    prefix: str = '',
    class_names: Optional[List[str]] = None
):
    """
    Print metrics in a nicely formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix string for printing
        class_names: List of class names
    """
    if class_names is None:
        class_names = ['none', 'weak', 'medium', 'strong']
    
    if prefix:
        print(f"\n{prefix}")
    
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    # Per-class metrics
    print(f"\n  Per-class metrics:")
    for class_name in class_names:
        p = metrics.get(f'precision_{class_name}', 0)
        r = metrics.get(f'recall_{class_name}', 0)
        f = metrics.get(f'f1_{class_name}', 0)
        print(f"    {class_name:>8}: P={p:.4f}, R={r:.4f}, F1={f:.4f}")
    
    # Confusion matrix
    if 'confusion_matrix' in metrics:
        print(f"\n  Confusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        
        # Header
        print(f"         ", end='')
        for class_name in class_names:
            print(f"{class_name:>8}", end=' ')
        print()
        
        # Rows
        for i, class_name in enumerate(class_names):
            print(f"    {class_name:>8}", end='')
            for j in range(len(class_names)):
                if i < cm.shape[0] and j < cm.shape[1]:
                    print(f" {cm[i][j]:>8}", end='')
                else:
                    print(f" {'0':>8}", end='')
            print()


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> str:
    """
    Get sklearn classification report as string.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        report: Classification report string
    """
    if class_names is None:
        class_names = ['none', 'weak', 'medium', 'strong']
    
    return classification_report(y_true, y_pred, target_names=class_names)


if __name__ == '__main__':
    """Test metrics."""
    print("="*80)
    print("Testing Metrics")
    print("="*80)
    
    # Create dummy predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 4, 100)
    y_pred = np.random.randint(0, 4, 100)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Print metrics
    print_metrics(metrics, "Test Metrics")
    
    # Per-class accuracy
    per_class_acc = calculate_per_class_accuracy(y_true, y_pred)
    print(f"\nPer-class Accuracy:")
    for class_name, acc in per_class_acc.items():
        print(f"  {class_name:>8}: {acc:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(get_classification_report(y_true, y_pred))
    
    print("\n" + "="*80)
    print("✓ Metrics test completed!")
    print("="*80)
