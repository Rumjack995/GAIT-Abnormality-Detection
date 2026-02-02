"""
Unit tests for the validation visualization system.

Tests cover performance charts, confusion matrices, training history plots,
model comparison visualizations, and interactive dashboards.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from gait_analysis.validation.visualization import ValidationVisualizer
from gait_analysis.utils.data_structures import (
    PerformanceMetrics, TrainingHistory, ModelComparison
)


class TestValidationVisualizer:
    """Test suite for ValidationVisualizer class."""
    
    @pytest.fixture
    def visualizer(self):
        """Create a ValidationVisualizer instance for testing."""
        return ValidationVisualizer(
            style='default',  # Use default style for testing
            figure_size=(10, 6),
            save_format='png'
        )
    
    @pytest.fixture
    def sample_performance_metrics(self):
        """Create sample performance metrics for testing."""
        return PerformanceMetrics(
            accuracy=0.85,
            precision={'normal': 0.90, 'limping': 0.80, 'shuffling': 0.75},
            recall={'normal': 0.88, 'limping': 0.82, 'shuffling': 0.78},
            f1_score={'normal': 0.89, 'limping': 0.81, 'shuffling': 0.76},
            training_time=120.0,
            inference_time=0.05,
            model_size=25.5
        )
    
    @pytest.fixture
    def sample_training_history(self):
        """Create sample training history for testing."""
        return TrainingHistory(
            train_loss=[0.8, 0.6, 0.4, 0.3, 0.25],
            val_loss=[0.9, 0.7, 0.5, 0.4, 0.35],
            train_accuracy=[0.6, 0.7, 0.8, 0.85, 0.88],
            val_accuracy=[0.55, 0.65, 0.75, 0.8, 0.83],
            epochs=5,
            best_epoch=4
        )
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization with different parameters."""
        # Test default initialization
        visualizer = ValidationVisualizer()
        assert visualizer.figure_size == (12, 8)
        assert visualizer.save_format == 'png'
        assert visualizer.dpi == 300
        
        # Test custom initialization
        visualizer = ValidationVisualizer(
            figure_size=(8, 6),
            save_format='jpg',
            dpi=150
        )
        assert visualizer.figure_size == (8, 6)
        assert visualizer.save_format == 'jpg'
        assert visualizer.dpi == 150
    
    def test_plot_performance_metrics(self, visualizer, sample_performance_metrics):
        """Test performance metrics plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_performance.png"
            
            # Create plot
            fig = visualizer.plot_performance_metrics(
                sample_performance_metrics,
                architecture_name="Test Model",
                save_path=str(save_path),
                show_plot=False
            )
            
            # Check that figure is created
            assert isinstance(fig, plt.Figure)
            assert len(fig.axes) == 4  # Should have 4 subplots
            
            # Check that file is saved
            assert save_path.exists()
            
            plt.close(fig)
    
    def test_plot_confusion_matrix(self, visualizer):
        """Test confusion matrix plotting."""
        # Create sample confusion matrix
        confusion_matrix = np.array([[15, 2, 1], [3, 18, 2], [1, 1, 17]])
        class_labels = ['normal', 'limping', 'shuffling']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_confusion.png"
            
            # Create plot
            fig = visualizer.plot_confusion_matrix(
                confusion_matrix,
                class_labels,
                architecture_name="Test Model",
                normalize=True,
                save_path=str(save_path),
                show_plot=False
            )
            
            # Check that figure is created
            assert isinstance(fig, plt.Figure)
            assert len(fig.axes) == 2  # Main plot + colorbar
            
            # Check that file is saved
            assert save_path.exists()
            
            plt.close(fig)
    
    def test_plot_training_history(self, visualizer, sample_training_history):
        """Test training history plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_training.png"
            
            # Create plot
            fig = visualizer.plot_training_history(
                sample_training_history,
                architecture_name="Test Model",
                save_path=str(save_path),
                show_plot=False
            )
            
            # Check that figure is created
            assert isinstance(fig, plt.Figure)
            assert len(fig.axes) == 2  # Loss and accuracy plots
            
            # Check that file is saved
            assert save_path.exists()
            
            plt.close(fig)
    
    def test_plot_model_comparison(self, visualizer, sample_performance_metrics, sample_training_history):
        """Test model comparison plotting."""
        # Create sample model comparisons
        model_comparisons = []
        architectures = ['3dcnn', 'lstm', 'hybrid']
        
        for i, arch in enumerate(architectures):
            # Vary metrics slightly for each architecture
            metrics = PerformanceMetrics(
                accuracy=sample_performance_metrics.accuracy + i * 0.02,
                precision={k: v + i * 0.01 for k, v in sample_performance_metrics.precision.items()},
                recall={k: v + i * 0.01 for k, v in sample_performance_metrics.recall.items()},
                f1_score={k: v + i * 0.01 for k, v in sample_performance_metrics.f1_score.items()},
                training_time=sample_performance_metrics.training_time + i * 10,
                inference_time=sample_performance_metrics.inference_time + i * 0.01,
                model_size=sample_performance_metrics.model_size + i * 5
            )
            
            comparison = ModelComparison(
                architecture_name=arch,
                performance_metrics=metrics,
                training_history=sample_training_history,
                model_path=f"models/{arch}_model.h5",
                hyperparameters={'batch_size': 32, 'epochs': 100}
            )
            model_comparisons.append(comparison)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_comparison.png"
            
            # Create plot
            fig = visualizer.plot_model_comparison(
                model_comparisons,
                save_path=str(save_path),
                show_plot=False
            )
            
            # Check that figure is created
            assert isinstance(fig, plt.Figure)
            assert len(fig.axes) == 6  # Should have 6 subplots
            
            # Check that file is saved
            assert save_path.exists()
            
            plt.close(fig)
    
    def test_plot_validation_trends(self, visualizer):
        """Test validation trends plotting."""
        # Create sample validation results
        validation_results = []
        architectures = ['3dcnn', 'lstm', 'hybrid']
        base_time = 1234567890
        
        for i in range(9):  # 3 results per architecture
            arch = architectures[i % 3]
            result = {
                'timestamp': base_time + i * 3600,  # 1 hour apart
                'architecture_type': arch,
                'performance_metrics': PerformanceMetrics(
                    accuracy=0.8 + (i % 3) * 0.02 + np.random.normal(0, 0.01),
                    precision={}, recall={}, f1_score={},
                    training_time=120, inference_time=0.05, model_size=25
                )
            }
            validation_results.append(result)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_trends.png"
            
            # Create plot
            fig = visualizer.plot_validation_trends(
                validation_results,
                save_path=str(save_path),
                show_plot=False
            )
            
            # Check that figure is created
            assert isinstance(fig, plt.Figure)
            assert len(fig.axes) == 1
            
            # Check that file is saved
            assert save_path.exists()
            
            plt.close(fig)
    
    @patch('gait_analysis.validation.visualization.pyo.plot')
    def test_create_interactive_dashboard(self, mock_plot, visualizer, sample_performance_metrics, sample_training_history):
        """Test interactive dashboard creation."""
        # Create sample model comparisons
        model_comparisons = []
        architectures = ['3dcnn', 'lstm']
        
        for i, arch in enumerate(architectures):
            metrics = PerformanceMetrics(
                accuracy=0.8 + i * 0.05,
                precision={'normal': 0.8, 'limping': 0.75},
                recall={'normal': 0.82, 'limping': 0.78},
                f1_score={'normal': 0.81, 'limping': 0.76},
                training_time=100 + i * 20,
                inference_time=0.05 + i * 0.01,
                model_size=20 + i * 10
            )
            
            comparison = ModelComparison(
                architecture_name=arch,
                performance_metrics=metrics,
                training_history=sample_training_history,
                model_path=f"models/{arch}_model.h5",
                hyperparameters={}
            )
            model_comparisons.append(comparison)
        
        # Create dashboard
        fig = visualizer.create_interactive_dashboard(model_comparisons)
        
        # Check that plotly figure is created
        assert fig is not None
        assert hasattr(fig, 'data')  # Plotly figure should have data attribute
    
    def test_save_all_plots(self, visualizer, sample_performance_metrics, sample_training_history):
        """Test saving all plots to directory."""
        # Create sample model comparisons
        model_comparisons = []
        architectures = ['3dcnn', 'lstm']
        
        for i, arch in enumerate(architectures):
            metrics = PerformanceMetrics(
                accuracy=0.8 + i * 0.05,
                precision={'normal': 0.8, 'limping': 0.75},
                recall={'normal': 0.82, 'limping': 0.78},
                f1_score={'normal': 0.81, 'limping': 0.76},
                training_time=100 + i * 20,
                inference_time=0.05 + i * 0.01,
                model_size=20 + i * 10
            )
            
            comparison = ModelComparison(
                architecture_name=arch,
                performance_metrics=metrics,
                training_history=sample_training_history,
                model_path=f"models/{arch}_model.h5",
                hyperparameters={}
            )
            model_comparisons.append(comparison)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save all plots
            visualizer.save_all_plots(temp_dir, model_comparisons)
            
            # Check that files are created
            output_dir = Path(temp_dir)
            assert (output_dir / "model_comparison.png").exists()
            assert (output_dir / "interactive_dashboard.html").exists()
            
            # Check individual model plots
            for arch in architectures:
                assert (output_dir / f"performance_{arch}.png").exists()
                assert (output_dir / f"training_history_{arch}.png").exists()
    
    def test_error_handling(self, visualizer):
        """Test error handling in visualization methods."""
        # Test with empty model comparisons
        with pytest.raises(ValueError, match="No model comparisons provided"):
            visualizer.plot_model_comparison([])
        
        with pytest.raises(ValueError, match="No model comparisons provided"):
            visualizer.create_interactive_dashboard([])
        
        # Test with empty validation results
        with pytest.raises(ValueError, match="No validation results provided"):
            visualizer.plot_validation_trends([])
    
    def test_color_schemes(self, visualizer):
        """Test that color schemes are properly defined."""
        assert 'primary' in visualizer.colors
        assert 'secondary' in visualizer.colors
        assert 'success' in visualizer.colors
        assert 'warning' in visualizer.colors
        
        # Colors should be valid hex codes
        for color_name, color_value in visualizer.colors.items():
            assert isinstance(color_value, str)
            assert color_value.startswith('#')
            assert len(color_value) == 7  # #RRGGBB format


if __name__ == '__main__':
    pytest.main([__file__])