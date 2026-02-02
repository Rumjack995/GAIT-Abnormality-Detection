"""
Performance visualization system for gait abnormality detection models.

This module provides comprehensive visualization capabilities including:
- Performance charts and validation visualizations
- Model comparison dashboards
- Training history and metrics visualization
- Interactive plots for detailed analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

from ..utils.data_structures import (
    PerformanceMetrics, TrainingHistory, ModelComparison
)


class ValidationVisualizer:
    """
    Comprehensive visualization system for model validation and performance analysis.
    
    This class provides various visualization capabilities for model performance,
    including static plots with matplotlib/seaborn and interactive dashboards
    with plotly for detailed analysis and model comparison.
    
    Features:
    - Performance metrics visualization (accuracy, precision, recall, F1)
    - Confusion matrix heatmaps with detailed analysis
    - Training history plots with loss and accuracy curves
    - Model comparison charts and dashboards
    - ROC and PR curves for classification analysis
    - Interactive dashboards for comprehensive model analysis
    """
    
    def __init__(self, 
                 style: str = 'seaborn-v0_8',
                 color_palette: str = 'husl',
                 figure_size: Tuple[int, int] = (12, 8),
                 save_format: str = 'png',
                 dpi: int = 300):
        """
        Initialize the visualization system.
        
        Args:
            style: Matplotlib style for plots
            color_palette: Seaborn color palette
            figure_size: Default figure size for plots
            save_format: Default format for saving plots
            dpi: Resolution for saved plots
        """
        self.style = style
        self.color_palette = color_palette
        self.figure_size = figure_size
        self.save_format = save_format
        self.dpi = dpi
        
        # Setup plotting style
        self._setup_plotting_style()
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Color schemes for different chart types
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'warning': '#C73E1D',
            'info': '#6A994E',
            'light': '#F5F5F5',
            'dark': '#2C3E50'
        }
        
    def _setup_plotting_style(self) -> None:
        """Setup matplotlib and seaborn plotting style."""
        # Set backend to Agg for headless environments (testing)
        import matplotlib
        if matplotlib.get_backend() != 'Agg':
            try:
                matplotlib.use('Agg')
            except Exception:
                pass  # Backend already set or cannot be changed
        
        try:
            plt.style.use(self.style)
        except OSError:
            # Fallback to default style if specified style is not available
            plt.style.use('default')
            
        sns.set_palette(self.color_palette)
        plt.rcParams['figure.figsize'] = self.figure_size
        plt.rcParams['savefig.dpi'] = self.dpi
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the visualizer."""
        logger = logging.getLogger('ValidationVisualizer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def plot_performance_metrics(self, 
                               performance_metrics: PerformanceMetrics,
                               architecture_name: str = "Model",
                               save_path: Optional[str] = None,
                               show_plot: bool = True) -> plt.Figure:
        """
        Create comprehensive performance metrics visualization.
        
        Args:
            performance_metrics: Performance metrics to visualize
            architecture_name: Name of the model architecture
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info(f"Creating performance metrics plot for {architecture_name}")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Performance Metrics - {architecture_name}', fontsize=16, fontweight='bold')
        
        # 1. Overall metrics bar chart
        ax1 = axes[0, 0]
        metrics_names = ['Accuracy', 'Avg Precision', 'Avg Recall', 'Avg F1-Score']
        metrics_values = [
            performance_metrics.accuracy,
            np.mean(list(performance_metrics.precision.values())),
            np.mean(list(performance_metrics.recall.values())),
            np.mean(list(performance_metrics.f1_score.values()))
        ]
        
        bars = ax1.bar(metrics_names, metrics_values, 
                      color=[self.colors['primary'], self.colors['secondary'], 
                            self.colors['success'], self.colors['info']])
        ax1.set_title('Overall Performance Metrics')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Per-class precision, recall, F1
        ax2 = axes[0, 1]
        class_labels = list(performance_metrics.precision.keys())
        x_pos = np.arange(len(class_labels))
        
        width = 0.25
        precision_values = list(performance_metrics.precision.values())
        recall_values = list(performance_metrics.recall.values())
        f1_values = list(performance_metrics.f1_score.values())
        
        ax2.bar(x_pos - width, precision_values, width, label='Precision', 
               color=self.colors['primary'], alpha=0.8)
        ax2.bar(x_pos, recall_values, width, label='Recall', 
               color=self.colors['secondary'], alpha=0.8)
        ax2.bar(x_pos + width, f1_values, width, label='F1-Score', 
               color=self.colors['success'], alpha=0.8)
        
        ax2.set_title('Per-Class Performance')
        ax2.set_ylabel('Score')
        ax2.set_xlabel('Classes')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(class_labels, rotation=45, ha='right')
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # 3. Performance vs Efficiency scatter
        ax3 = axes[1, 0]
        efficiency_metrics = {
            'Inference Time (ms)': performance_metrics.inference_time * 1000,
            'Model Size (MB)': performance_metrics.model_size
        }
        
        # Create scatter plot showing accuracy vs efficiency
        ax3.scatter([performance_metrics.inference_time * 1000], 
                   [performance_metrics.accuracy], 
                   s=performance_metrics.model_size * 10, 
                   color=self.colors['warning'], alpha=0.7)
        
        ax3.set_xlabel('Inference Time (ms)')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Performance vs Efficiency')
        ax3.grid(True, alpha=0.3)
        
        # Add annotation
        ax3.annotate(f'{architecture_name}\nSize: {performance_metrics.model_size:.1f}MB',
                    xy=(performance_metrics.inference_time * 1000, performance_metrics.accuracy),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # 4. Metrics summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary table data
        table_data = [
            ['Metric', 'Value'],
            ['Accuracy', f'{performance_metrics.accuracy:.4f}'],
            ['Avg Precision', f'{np.mean(list(performance_metrics.precision.values())):.4f}'],
            ['Avg Recall', f'{np.mean(list(performance_metrics.recall.values())):.4f}'],
            ['Avg F1-Score', f'{np.mean(list(performance_metrics.f1_score.values())):.4f}'],
            ['Inference Time', f'{performance_metrics.inference_time*1000:.2f} ms'],
            ['Model Size', f'{performance_metrics.model_size:.2f} MB'],
            ['Training Time', f'{performance_metrics.training_time:.2f} s']
        ]
        
        table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc='center', loc='center',
                         colWidths=[0.4, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_data)):
            if i == 0:  # Header
                table[(i, 0)].set_facecolor(self.colors['primary'])
                table[(i, 1)].set_facecolor(self.colors['primary'])
                table[(i, 0)].set_text_props(weight='bold', color='white')
                table[(i, 1)].set_text_props(weight='bold', color='white')
            else:
                table[(i, 0)].set_facecolor(self.colors['light'])
                table[(i, 1)].set_facecolor('white')
        
        ax4.set_title('Performance Summary')
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Performance metrics plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_confusion_matrix(self, 
                            confusion_matrix_data: np.ndarray,
                            class_labels: List[str],
                            architecture_name: str = "Model",
                            normalize: bool = True,
                            save_path: Optional[str] = None,
                            show_plot: bool = True) -> plt.Figure:
        """
        Create confusion matrix heatmap visualization.
        
        Args:
            confusion_matrix_data: Confusion matrix as numpy array
            class_labels: List of class labels
            architecture_name: Name of the model architecture
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info(f"Creating confusion matrix plot for {architecture_name}")
        
        # Normalize if requested
        if normalize:
            cm_normalized = confusion_matrix_data.astype('float') / confusion_matrix_data.sum(axis=1)[:, np.newaxis]
            cm_display = cm_normalized
            fmt = '.2f'
            title_suffix = ' (Normalized)'
        else:
            cm_display = confusion_matrix_data
            fmt = 'd'
            title_suffix = ''
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm_display, 
                   annot=True, 
                   fmt=fmt, 
                   cmap='Blues',
                   xticklabels=class_labels,
                   yticklabels=class_labels,
                   ax=ax,
                   cbar_kws={'label': 'Normalized Count' if normalize else 'Count'})
        
        ax.set_title(f'Confusion Matrix - {architecture_name}{title_suffix}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Add performance statistics as text
        if normalize:
            # Calculate per-class accuracy from normalized confusion matrix
            class_accuracies = np.diag(cm_normalized)
            overall_accuracy = np.trace(confusion_matrix_data) / np.sum(confusion_matrix_data)
            
            stats_text = f'Overall Accuracy: {overall_accuracy:.3f}\n'
            stats_text += 'Per-class Accuracy:\n'
            for i, (label, acc) in enumerate(zip(class_labels, class_accuracies)):
                stats_text += f'  {label}: {acc:.3f}\n'
            
            # Add text box with statistics
            ax.text(1.02, 0.5, stats_text, transform=ax.transAxes, 
                   verticalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Confusion matrix plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_training_history(self, 
                            training_history: TrainingHistory,
                            architecture_name: str = "Model",
                            save_path: Optional[str] = None,
                            show_plot: bool = True) -> plt.Figure:
        """
        Create training history visualization with loss and accuracy curves.
        
        Args:
            training_history: Training history data
            architecture_name: Name of the model architecture
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info(f"Creating training history plot for {architecture_name}")
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Training History - {architecture_name}', fontsize=16, fontweight='bold')
        
        epochs = range(1, training_history.epochs + 1)
        
        # Plot training and validation loss
        ax1.plot(epochs, training_history.train_loss, 
                label='Training Loss', color=self.colors['primary'], linewidth=2)
        ax1.plot(epochs, training_history.val_loss, 
                label='Validation Loss', color=self.colors['secondary'], linewidth=2)
        
        # Mark best epoch
        if hasattr(training_history, 'best_epoch') and training_history.best_epoch > 0:
            ax1.axvline(x=training_history.best_epoch, color=self.colors['warning'], 
                       linestyle='--', alpha=0.7, label=f'Best Epoch ({training_history.best_epoch})')
        
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot training and validation accuracy
        ax2.plot(epochs, training_history.train_accuracy, 
                label='Training Accuracy', color=self.colors['success'], linewidth=2)
        ax2.plot(epochs, training_history.val_accuracy, 
                label='Validation Accuracy', color=self.colors['info'], linewidth=2)
        
        # Mark best epoch
        if hasattr(training_history, 'best_epoch') and training_history.best_epoch > 0:
            ax2.axvline(x=training_history.best_epoch, color=self.colors['warning'], 
                       linestyle='--', alpha=0.7, label=f'Best Epoch ({training_history.best_epoch})')
        
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add final performance annotations
        final_train_acc = training_history.train_accuracy[-1]
        final_val_acc = training_history.val_accuracy[-1]
        final_train_loss = training_history.train_loss[-1]
        final_val_loss = training_history.val_loss[-1]
        
        ax1.annotate(f'Final: {final_val_loss:.4f}', 
                    xy=(training_history.epochs, final_val_loss),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax2.annotate(f'Final: {final_val_acc:.4f}', 
                    xy=(training_history.epochs, final_val_acc),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Training history plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_model_comparison(self, 
                            model_comparisons: List[ModelComparison],
                            save_path: Optional[str] = None,
                            show_plot: bool = True) -> plt.Figure:
        """
        Create comprehensive model comparison visualization.
        
        Args:
            model_comparisons: List of model comparison objects
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info(f"Creating model comparison plot for {len(model_comparisons)} models")
        
        if not model_comparisons:
            raise ValueError("No model comparisons provided")
        
        # Extract data for comparison
        architectures = [comp.architecture_name for comp in model_comparisons]
        accuracies = [comp.performance_metrics.accuracy for comp in model_comparisons]
        avg_precisions = [np.mean(list(comp.performance_metrics.precision.values())) 
                         for comp in model_comparisons]
        avg_recalls = [np.mean(list(comp.performance_metrics.recall.values())) 
                      for comp in model_comparisons]
        avg_f1s = [np.mean(list(comp.performance_metrics.f1_score.values())) 
                  for comp in model_comparisons]
        inference_times = [comp.performance_metrics.inference_time * 1000 
                          for comp in model_comparisons]
        model_sizes = [comp.performance_metrics.model_size for comp in model_comparisons]
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Architecture Comparison', fontsize=16, fontweight='bold')
        
        # 1. Performance metrics comparison
        ax1 = axes[0, 0]
        x_pos = np.arange(len(architectures))
        width = 0.2
        
        ax1.bar(x_pos - 1.5*width, accuracies, width, label='Accuracy', 
               color=self.colors['primary'], alpha=0.8)
        ax1.bar(x_pos - 0.5*width, avg_precisions, width, label='Precision', 
               color=self.colors['secondary'], alpha=0.8)
        ax1.bar(x_pos + 0.5*width, avg_recalls, width, label='Recall', 
               color=self.colors['success'], alpha=0.8)
        ax1.bar(x_pos + 1.5*width, avg_f1s, width, label='F1-Score', 
               color=self.colors['info'], alpha=0.8)
        
        ax1.set_title('Performance Metrics Comparison')
        ax1.set_ylabel('Score')
        ax1.set_xlabel('Architecture')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(architectures, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # 2. Inference time comparison
        ax2 = axes[0, 1]
        bars = ax2.bar(architectures, inference_times, color=self.colors['warning'], alpha=0.8)
        ax2.set_title('Inference Time Comparison')
        ax2.set_ylabel('Time (ms)')
        ax2.set_xlabel('Architecture')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, inference_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(inference_times)*0.01,
                    f'{value:.1f}ms', ha='center', va='bottom')
        
        # 3. Model size comparison
        ax3 = axes[0, 2]
        bars = ax3.bar(architectures, model_sizes, color=self.colors['dark'], alpha=0.8)
        ax3.set_title('Model Size Comparison')
        ax3.set_ylabel('Size (MB)')
        ax3.set_xlabel('Architecture')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, model_sizes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(model_sizes)*0.01,
                    f'{value:.1f}MB', ha='center', va='bottom')
        
        # 4. Performance vs Efficiency scatter plot
        ax4 = axes[1, 0]
        scatter = ax4.scatter(inference_times, accuracies, 
                            s=[size*10 for size in model_sizes], 
                            c=range(len(architectures)), 
                            cmap='viridis', alpha=0.7)
        
        # Add labels for each point
        for i, arch in enumerate(architectures):
            ax4.annotate(arch, (inference_times[i], accuracies[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        ax4.set_xlabel('Inference Time (ms)')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Performance vs Efficiency\n(Bubble size = Model size)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Radar chart for overall comparison
        ax5 = axes[1, 1]
        
        # Normalize metrics for radar chart (0-1 scale)
        normalized_metrics = []
        for i in range(len(architectures)):
            metrics = [
                accuracies[i],
                avg_precisions[i],
                avg_recalls[i],
                avg_f1s[i],
                1 - (inference_times[i] / max(inference_times)),  # Invert for better visualization
                1 - (model_sizes[i] / max(model_sizes))  # Invert for better visualization
            ]
            normalized_metrics.append(metrics)
        
        # Create radar chart data
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Speed', 'Efficiency']
        
        # Plot radar chart for first model (simplified version)
        if len(architectures) > 0:
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            for i, (arch, metrics) in enumerate(zip(architectures, normalized_metrics)):
                values = metrics + metrics[:1]  # Complete the circle
                ax5.plot(angles, values, 'o-', linewidth=2, label=arch)
                ax5.fill(angles, values, alpha=0.25)
            
            ax5.set_xticks(angles[:-1])
            ax5.set_xticklabels(categories)
            ax5.set_ylim(0, 1)
            ax5.set_title('Overall Performance Radar')
            ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax5.grid(True)
        
        # 6. Summary table
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create summary table
        table_data = [['Architecture', 'Accuracy', 'F1-Score', 'Time (ms)', 'Size (MB)']]
        for i, arch in enumerate(architectures):
            table_data.append([
                arch,
                f'{accuracies[i]:.3f}',
                f'{avg_f1s[i]:.3f}',
                f'{inference_times[i]:.1f}',
                f'{model_sizes[i]:.1f}'
            ])
        
        table = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style the table
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                if i == 0:  # Header
                    table[(i, j)].set_facecolor(self.colors['primary'])
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    table[(i, j)].set_facecolor(self.colors['light'] if i % 2 == 0 else 'white')
        
        ax6.set_title('Performance Summary')
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Model comparison plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_interactive_dashboard(self, 
                                   model_comparisons: List[ModelComparison],
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive dashboard using Plotly for comprehensive model analysis.
        
        Args:
            model_comparisons: List of model comparison objects
            save_path: Path to save the HTML dashboard (optional)
            
        Returns:
            Plotly figure object
        """
        self.logger.info(f"Creating interactive dashboard for {len(model_comparisons)} models")
        
        if not model_comparisons:
            raise ValueError("No model comparisons provided")
        
        # Extract data
        architectures = [comp.architecture_name for comp in model_comparisons]
        performance_data = []
        
        for comp in model_comparisons:
            metrics = comp.performance_metrics
            performance_data.append({
                'Architecture': comp.architecture_name,
                'Accuracy': metrics.accuracy,
                'Avg_Precision': np.mean(list(metrics.precision.values())),
                'Avg_Recall': np.mean(list(metrics.recall.values())),
                'Avg_F1': np.mean(list(metrics.f1_score.values())),
                'Inference_Time_ms': metrics.inference_time * 1000,
                'Model_Size_MB': metrics.model_size,
                'Training_Time_s': metrics.training_time
            })
        
        df = pd.DataFrame(performance_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Metrics', 'Efficiency Analysis', 
                          'Training History', 'Model Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Performance metrics bar chart
        fig.add_trace(
            go.Bar(name='Accuracy', x=df['Architecture'], y=df['Accuracy'],
                  marker_color='#2E86AB', showlegend=True),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='F1-Score', x=df['Architecture'], y=df['Avg_F1'],
                  marker_color='#A23B72', showlegend=True),
            row=1, col=1
        )
        
        # 2. Efficiency scatter plot
        fig.add_trace(
            go.Scatter(
                x=df['Inference_Time_ms'], 
                y=df['Accuracy'],
                mode='markers+text',
                marker=dict(size=df['Model_Size_MB']*2, opacity=0.7, color='#F18F01'),
                text=df['Architecture'],
                textposition="top center",
                name='Efficiency',
                showlegend=True
            ),
            row=1, col=2
        )
        
        # 3. Model size comparison
        fig.add_trace(
            go.Bar(name='Model Size', x=df['Architecture'], y=df['Model_Size_MB'],
                  marker_color='#C73E1D', showlegend=True),
            row=2, col=1
        )
        
        # 4. Performance radar chart (simplified as bar chart for subplot)
        fig.add_trace(
            go.Bar(name='Inference Time', x=df['Architecture'], y=df['Inference_Time_ms'],
                  marker_color='#6A994E', showlegend=True),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Model Performance Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Architecture", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        
        fig.update_xaxes(title_text="Inference Time (ms)", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        
        fig.update_xaxes(title_text="Architecture", row=2, col=1)
        fig.update_yaxes(title_text="Size (MB)", row=2, col=1)
        
        fig.update_xaxes(title_text="Architecture", row=2, col=2)
        fig.update_yaxes(title_text="Time (ms)", row=2, col=2)
        
        # Save interactive dashboard if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if save_path.suffix == '.html':
                pyo.plot(fig, filename=str(save_path), auto_open=False)
            else:
                # Save as static image
                fig.write_image(str(save_path))
            
            self.logger.info(f"Interactive dashboard saved to {save_path}")
        
        return fig
    
    def plot_validation_trends(self, 
                             validation_results: List[Dict[str, Any]],
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> plt.Figure:
        """
        Plot validation performance trends over time.
        
        Args:
            validation_results: List of validation result dictionaries
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info(f"Creating validation trends plot for {len(validation_results)} results")
        
        if not validation_results:
            raise ValueError("No validation results provided")
        
        # Extract data
        timestamps = [result['timestamp'] for result in validation_results]
        architectures = [result['architecture_type'] for result in validation_results]
        accuracies = [result['performance_metrics'].accuracy for result in validation_results]
        
        # Convert timestamps to relative time (hours from first validation)
        base_time = min(timestamps)
        relative_times = [(ts - base_time) / 3600 for ts in timestamps]  # Convert to hours
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot trends for each architecture
        unique_architectures = list(set(architectures))
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_architectures)))
        
        for i, arch in enumerate(unique_architectures):
            arch_indices = [j for j, a in enumerate(architectures) if a == arch]
            arch_times = [relative_times[j] for j in arch_indices]
            arch_accuracies = [accuracies[j] for j in arch_indices]
            
            ax.plot(arch_times, arch_accuracies, 'o-', 
                   color=colors[i], label=arch, linewidth=2, markersize=6)
        
        ax.set_title('Validation Performance Trends Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (hours from first validation)')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations for best performance
        best_idx = np.argmax(accuracies)
        ax.annotate(f'Best: {accuracies[best_idx]:.4f}\n{architectures[best_idx]}',
                   xy=(relative_times[best_idx], accuracies[best_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Validation trends plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def save_all_plots(self, 
                      output_dir: str,
                      model_comparisons: List[ModelComparison],
                      validation_results: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Generate and save all visualization plots to a directory.
        
        Args:
            output_dir: Directory to save all plots
            model_comparisons: List of model comparison objects
            validation_results: Optional validation results for trends
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating all plots in directory: {output_dir}")
        
        # 1. Individual performance plots for each model
        for comp in model_comparisons:
            self.plot_performance_metrics(
                comp.performance_metrics,
                comp.architecture_name,
                save_path=output_dir / f"performance_{comp.architecture_name}.{self.save_format}",
                show_plot=False
            )
            
            # Training history if available
            if comp.training_history:
                self.plot_training_history(
                    comp.training_history,
                    comp.architecture_name,
                    save_path=output_dir / f"training_history_{comp.architecture_name}.{self.save_format}",
                    show_plot=False
                )
        
        # 2. Model comparison plot
        self.plot_model_comparison(
            model_comparisons,
            save_path=output_dir / f"model_comparison.{self.save_format}",
            show_plot=False
        )
        
        # 3. Interactive dashboard
        self.create_interactive_dashboard(
            model_comparisons,
            save_path=output_dir / "interactive_dashboard.html"
        )
        
        # 4. Validation trends if available
        if validation_results:
            self.plot_validation_trends(
                validation_results,
                save_path=output_dir / f"validation_trends.{self.save_format}",
                show_plot=False
            )
        
        self.logger.info(f"All plots saved to {output_dir}")