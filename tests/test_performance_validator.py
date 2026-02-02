"""
Unit tests for the performance validation system.

Tests cover performance validation pipeline, threshold checking,
model comparison, and recommendation generation.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from gait_analysis.validation.performance_validator import PerformanceValidator
from gait_analysis.utils.data_structures import (
    PerformanceMetrics, Dataset, TrainingExample, 
    ModelComparison, PoseSequence, PoseKeypoint
)
from gait_analysis.classification.gait_classifier import GaitClassifier


class TestPerformanceValidator:
    """Test suite for PerformanceValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a PerformanceValidator instance for testing."""
        return PerformanceValidator(
            validation_mode='moderate',
            class_labels=['normal', 'limping', 'shuffling']
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
    def sample_test_dataset(self):
        """Create a sample test dataset for validation."""
        examples = []
        labels = ['normal', 'limping', 'shuffling']
        
        for i in range(60):  # 20 samples per class
            label = labels[i % 3]
            
            # Create dummy pose sequence
            keypoints = []
            for frame in range(10):
                frame_keypoints = []
                for kp in range(33):  # MediaPipe has 33 keypoints
                    frame_keypoints.append(PoseKeypoint(
                        x=np.random.rand(),
                        y=np.random.rand(),
                        z=np.random.rand(),
                        confidence=0.8 + np.random.rand() * 0.2
                    ))
                keypoints.append(frame_keypoints)
            
            pose_sequence = PoseSequence(
                keypoints=keypoints,
                timestamps=list(range(10)),
                confidence_scores=[0.9] * 10
            )
            
            example = TrainingExample(
                video_path=f"test_video_{i}.mp4",
                pose_sequence=pose_sequence,
                ground_truth_label=label,
                severity_score=0.5,
                metadata={'test': True, 'features': np.random.randn(16, 224, 224, 3)}
            )
            examples.append(example)
        
        return Dataset(
            examples=examples,
            class_distribution={'normal': 20, 'limping': 20, 'shuffling': 20},
            validation_split=0.2
        )
    
    @pytest.fixture
    def mock_classifier(self):
        """Create a mock classifier for testing."""
        classifier = Mock(spec=GaitClassifier)
        classifier.architecture_type = 'hybrid'
        classifier.class_labels = ['normal', 'limping', 'shuffling']
        classifier.model = Mock()
        classifier.model.count_params.return_value = 1000000  # 1M parameters
        
        # Mock prediction method
        def mock_predict(input_data, return_probabilities=False, return_uncertainty=False):
            if return_probabilities:
                return {
                    'classification_result': Mock(abnormality_type='normal'),
                    'class_probabilities': {
                        'normal': 0.8,
                        'limping': 0.15,
                        'shuffling': 0.05
                    }
                }
            else:
                return Mock(abnormality_type='normal')
        
        classifier.predict = mock_predict
        classifier.predict_proba = lambda x: {'normal': 0.8, 'limping': 0.15, 'shuffling': 0.05}
        
        return classifier
    
    def test_validator_initialization(self):
        """Test validator initialization with different parameters."""
        # Test default initialization
        validator = PerformanceValidator()
        assert validator.validation_mode == 'strict'
        assert len(validator.class_labels) > 0
        assert validator.thresholds['accuracy'] > 0
        
        # Test custom initialization
        custom_thresholds = {'accuracy': 0.9, 'precision': 0.85}
        validator = PerformanceValidator(
            performance_thresholds=custom_thresholds,
            validation_mode='lenient'
        )
        # Lenient mode adjusts thresholds by 0.9, so 0.9 * 0.9 = 0.81
        assert validator.thresholds['accuracy'] == 0.81
        assert validator.validation_mode == 'lenient'
    
    def test_threshold_adjustment_by_mode(self):
        """Test threshold adjustment based on validation mode."""
        # Strict mode should have higher thresholds
        strict_validator = PerformanceValidator(validation_mode='strict')
        moderate_validator = PerformanceValidator(validation_mode='moderate')
        lenient_validator = PerformanceValidator(validation_mode='lenient')
        
        # Compare accuracy thresholds
        assert strict_validator.thresholds['accuracy'] > moderate_validator.thresholds['accuracy']
        assert moderate_validator.thresholds['accuracy'] > lenient_validator.thresholds['accuracy']
    
    def test_validate_test_dataset(self, validator, sample_test_dataset):
        """Test test dataset validation."""
        # Valid dataset should pass
        validator._validate_test_dataset(sample_test_dataset)
        
        # Empty dataset should fail
        empty_dataset = Dataset(examples=[], class_distribution={}, validation_split=0.2)
        with pytest.raises(ValueError, match="Test dataset is empty"):
            validator._validate_test_dataset(empty_dataset)
        
        # Small dataset should fail
        small_dataset = Dataset(
            examples=sample_test_dataset.examples[:10],
            class_distribution={'normal': 5, 'limping': 5},
            validation_split=0.2
        )
        with pytest.raises(ValueError, match="Test dataset too small"):
            validator._validate_test_dataset(small_dataset)
    
    def test_prepare_test_data(self, validator, sample_test_dataset):
        """Test test data preparation."""
        X_test, y_test, y_test_encoded = validator._prepare_test_data(sample_test_dataset)
        
        assert len(X_test) == len(sample_test_dataset.examples)
        assert len(y_test) == len(sample_test_dataset.examples)
        assert len(y_test_encoded) == len(sample_test_dataset.examples)
        
        # Check that labels are properly encoded
        unique_labels = set(y_test)
        assert unique_labels == {'normal', 'limping', 'shuffling'}
        
        # Check encoded labels are valid indices
        assert all(0 <= idx < len(validator.class_labels) for idx in y_test_encoded)
    
    def test_calculate_model_size(self, validator, mock_classifier):
        """Test model size calculation."""
        size_mb = validator._calculate_model_size(mock_classifier)
        
        # Should calculate size based on parameter count
        assert size_mb > 0
        assert isinstance(size_mb, float)
        
        # Test with no model
        mock_classifier.model = None
        size_mb = validator._calculate_model_size(mock_classifier)
        assert size_mb == 0.0
    
    def test_validate_model_performance(self, validator, mock_classifier, sample_test_dataset):
        """Test model performance validation."""
        # Run validation
        performance_metrics = validator.validate_model_performance(
            mock_classifier, 
            sample_test_dataset,
            calculate_auc=False,  # Skip AUC for simpler testing
            save_results=True
        )
        
        # Check that performance metrics are returned
        assert isinstance(performance_metrics, PerformanceMetrics)
        assert 0 <= performance_metrics.accuracy <= 1
        assert performance_metrics.inference_time >= 0  # Can be 0 for very fast inference
        assert performance_metrics.model_size >= 0
        
        # Check that results are saved
        assert len(validator.validation_results) == 1
        
        # Check per-class metrics
        assert len(performance_metrics.precision) == len(validator.class_labels)
        assert len(performance_metrics.recall) == len(validator.class_labels)
        assert len(performance_metrics.f1_score) == len(validator.class_labels)
    
    def test_check_performance_thresholds(self, validator, sample_performance_metrics):
        """Test performance threshold checking."""
        # Test with good performance (should pass all thresholds)
        results = validator.check_performance_thresholds(sample_performance_metrics)
        
        assert isinstance(results, dict)
        assert 'meets_thresholds' in results
        assert 'failed_metrics' in results
        assert 'recommendations' in results
        assert 'threshold_analysis' in results
        
        # Check threshold analysis structure
        assert 'accuracy' in results['threshold_analysis']
        assert 'precision' in results['threshold_analysis']
        assert 'recall' in results['threshold_analysis']
        
        # Test with poor performance
        poor_metrics = PerformanceMetrics(
            accuracy=0.5,  # Below threshold
            precision={'normal': 0.4, 'limping': 0.3, 'shuffling': 0.2},
            recall={'normal': 0.4, 'limping': 0.3, 'shuffling': 0.2},
            f1_score={'normal': 0.4, 'limping': 0.3, 'shuffling': 0.2},
            training_time=120.0,
            inference_time=0.2,  # Slow inference
            model_size=100.0  # Large model
        )
        
        results = validator.check_performance_thresholds(poor_metrics)
        assert not results['meets_thresholds']
        assert len(results['failed_metrics']) > 0
        assert len(results['recommendations']) > 0
    
    def test_recommend_model_improvements(self, validator, sample_performance_metrics):
        """Test model improvement recommendations."""
        # Create threshold results
        threshold_results = validator.check_performance_thresholds(sample_performance_metrics)
        
        # Get recommendations
        recommendations = validator.recommend_model_improvements(
            sample_performance_metrics, 
            threshold_results
        )
        
        assert isinstance(recommendations, list)
        
        # Test with poor performance metrics
        poor_metrics = PerformanceMetrics(
            accuracy=0.4,
            precision={'normal': 0.9, 'limping': 0.2, 'shuffling': 0.1},  # High variance
            recall={'normal': 0.3, 'limping': 0.9, 'shuffling': 0.2},
            f1_score={'normal': 0.4, 'limping': 0.3, 'shuffling': 0.1},
            training_time=120.0,
            inference_time=0.6,  # Very slow
            model_size=150.0  # Very large
        )
        
        threshold_results = validator.check_performance_thresholds(poor_metrics)
        recommendations = validator.recommend_model_improvements(poor_metrics, threshold_results)
        
        assert len(recommendations) > 0
        # Should detect high variance and poor overall performance
        recommendation_text = ' '.join(recommendations).lower()
        assert any(keyword in recommendation_text for keyword in ['variance', 'accuracy', 'slow', 'large'])
    
    def test_compare_models(self, validator):
        """Test model comparison functionality."""
        # Create sample model comparisons
        model_comparisons = []
        architectures = ['3dcnn', 'lstm', 'hybrid']
        
        for i, arch in enumerate(architectures):
            metrics = PerformanceMetrics(
                accuracy=0.8 + i * 0.05,
                precision={'normal': 0.8 + i * 0.03, 'limping': 0.75 + i * 0.03, 'shuffling': 0.7 + i * 0.03},
                recall={'normal': 0.82 + i * 0.02, 'limping': 0.78 + i * 0.02, 'shuffling': 0.72 + i * 0.02},
                f1_score={'normal': 0.81 + i * 0.025, 'limping': 0.76 + i * 0.025, 'shuffling': 0.71 + i * 0.025},
                training_time=100.0 + i * 20,
                inference_time=0.05 + i * 0.01,
                model_size=20.0 + i * 10
            )
            
            comparison = ModelComparison(
                architecture_name=arch,
                performance_metrics=metrics,
                training_history=Mock(),  # Mock training history
                model_path=f"models/{arch}_model.h5",
                hyperparameters={'batch_size': 32, 'epochs': 100}
            )
            model_comparisons.append(comparison)
        
        # Run comparison
        results = validator.compare_models(model_comparisons)
        
        assert isinstance(results, dict)
        assert 'model_count' in results
        assert 'comparison_data' in results
        assert 'best_models' in results
        assert 'best_overall' in results
        assert 'recommendations' in results
        
        assert results['model_count'] == 3
        assert len(results['comparison_data']) == 3
        assert 'accuracy' in results['best_models']
        assert results['best_overall'] in architectures
        
        # Test with empty list
        with pytest.raises(ValueError, match="No model comparisons provided"):
            validator.compare_models([])
    
    def test_save_and_load_validation_results(self, validator, sample_performance_metrics):
        """Test saving and loading validation results."""
        # Add some validation results
        validator.validation_results = [{
            'timestamp': 1234567890,
            'architecture_type': 'hybrid',
            'performance_metrics': sample_performance_metrics,
            'additional_metrics': {},
            'classification_report': {},
            'confusion_matrix': [[10, 2], [1, 12]],
            'test_samples': 25,
            'class_distribution': {0: 12, 1: 13}
        }]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test saving
            validator.save_validation_results(temp_path)
            assert Path(temp_path).exists()
            
            # Test loading
            new_validator = PerformanceValidator()
            new_validator.load_validation_results(temp_path)
            
            assert len(new_validator.validation_results) == 1
            assert new_validator.validation_results[0]['architecture_type'] == 'hybrid'
            
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
        
        # Test loading non-existent file
        with pytest.raises(FileNotFoundError):
            validator.load_validation_results('non_existent_file.json')
    
    def test_get_validation_summary(self, validator):
        """Test validation summary generation."""
        # Test with no results
        summary = validator.get_validation_summary()
        assert 'message' in summary
        assert summary['message'] == 'No validation results available'
        
        # Add some validation results
        validator.validation_results = [
            {
                'timestamp': 1234567890,
                'architecture_type': 'hybrid',
                'performance_metrics': PerformanceMetrics(
                    accuracy=0.85, precision={}, recall={}, f1_score={},
                    training_time=120, inference_time=0.05, model_size=25
                )
            },
            {
                'timestamp': 1234567900,
                'architecture_type': '3dcnn',
                'performance_metrics': PerformanceMetrics(
                    accuracy=0.82, precision={}, recall={}, f1_score={},
                    training_time=150, inference_time=0.08, model_size=30
                )
            }
        ]
        
        summary = validator.get_validation_summary()
        
        assert summary['total_validations'] == 2
        assert len(summary['architectures_tested']) == 2
        assert 'hybrid' in summary['architectures_tested']
        assert '3dcnn' in summary['architectures_tested']
        assert 'average_metrics' in summary
        assert 'best_performances' in summary
        assert summary['average_metrics']['accuracy'] == 0.835  # (0.85 + 0.82) / 2
    
    def test_edge_cases(self, validator):
        """Test edge cases and error handling."""
        # Test with invalid validation mode - should not raise error during initialization
        validator_invalid = PerformanceValidator(validation_mode='invalid_mode')
        assert validator_invalid.validation_mode == 'invalid_mode'
        
        # Test with empty class labels - should use default labels
        validator_empty = PerformanceValidator(class_labels=[])
        assert len(validator_empty.class_labels) > 0  # Should use default labels
        
        # Test threshold checking with missing metrics
        incomplete_metrics = PerformanceMetrics(
            accuracy=0.8,
            precision={},  # Empty precision dict
            recall={},     # Empty recall dict
            f1_score={},   # Empty f1_score dict
            training_time=120,
            inference_time=0.05,
            model_size=25
        )
        
        # Should handle empty metric dictionaries gracefully
        results = validator.check_performance_thresholds(incomplete_metrics)
        assert isinstance(results, dict)
        assert 'meets_thresholds' in results


if __name__ == '__main__':
    pytest.main([__file__])