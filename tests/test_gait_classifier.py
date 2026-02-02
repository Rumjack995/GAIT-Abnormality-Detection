"""
Unit tests for GaitClassifier class.

Tests the multi-architecture gait classification system including model loading,
inference, confidence scoring, and uncertainty quantification.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from gait_analysis.classification.gait_classifier import GaitClassifier
from gait_analysis.utils.data_structures import ClassificationResult


class TestGaitClassifier:
    """Test suite for GaitClassifier class."""
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        classifier = GaitClassifier()
        
        assert classifier.architecture_type == 'hybrid'
        assert classifier.class_labels == GaitClassifier.DEFAULT_CLASS_LABELS
        assert classifier.num_classes == len(GaitClassifier.DEFAULT_CLASS_LABELS)
        assert classifier.confidence_threshold == 0.7
        assert classifier.uncertainty_threshold == 0.3
        assert classifier.model is None
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_labels = ['normal', 'abnormal']
        classifier = GaitClassifier(
            architecture_type='3dcnn',
            class_labels=custom_labels,
            confidence_threshold=0.8,
            uncertainty_threshold=0.2
        )
        
        assert classifier.architecture_type == '3dcnn'
        assert classifier.class_labels == custom_labels
        assert classifier.num_classes == 2
        assert classifier.confidence_threshold == 0.8
        assert classifier.uncertainty_threshold == 0.2
    
    def test_init_invalid_architecture(self):
        """Test initialization with invalid architecture type."""
        with pytest.raises(ValueError, match="Unsupported architecture"):
            GaitClassifier(architecture_type='invalid_arch')
    
    @patch('gait_analysis.classification.gait_classifier.create_hybrid_cnn_lstm')
    def test_build_model_hybrid(self, mock_create_hybrid):
        """Test building hybrid CNN-LSTM model."""
        # Mock the model creation
        mock_wrapper = Mock()
        mock_model = Mock()
        mock_wrapper.model = mock_model
        mock_model.count_params.return_value = 1000000
        mock_create_hybrid.return_value = mock_wrapper
        
        classifier = GaitClassifier(architecture_type='hybrid')
        classifier.build_model()
        
        assert classifier.model_wrapper == mock_wrapper
        assert classifier.model == mock_model
        mock_create_hybrid.assert_called_once()
    
    @patch('gait_analysis.classification.gait_classifier.create_lightweight_3dcnn')
    def test_build_model_3dcnn(self, mock_create_3dcnn):
        """Test building 3D-CNN model."""
        # Mock the model creation
        mock_wrapper = Mock()
        mock_model = Mock()
        mock_wrapper.model = mock_model
        mock_model.count_params.return_value = 500000
        mock_create_3dcnn.return_value = mock_wrapper
        
        classifier = GaitClassifier(architecture_type='3dcnn')
        classifier.build_model()
        
        assert classifier.model_wrapper == mock_wrapper
        assert classifier.model == mock_model
        mock_create_3dcnn.assert_called_once()
    
    @patch('gait_analysis.classification.gait_classifier.create_efficient_lstm')
    def test_build_model_lstm(self, mock_create_lstm):
        """Test building LSTM model."""
        # Mock the model creation
        mock_wrapper = Mock()
        mock_model = Mock()
        mock_wrapper.model = mock_model
        mock_model.count_params.return_value = 200000
        mock_create_lstm.return_value = mock_wrapper
        
        classifier = GaitClassifier(architecture_type='lstm')
        classifier.build_model()
        
        assert classifier.model_wrapper == mock_wrapper
        assert classifier.model == mock_model
        mock_create_lstm.assert_called_once()
    
    def test_predict_no_model_loaded(self):
        """Test prediction without loaded model raises error."""
        classifier = GaitClassifier()
        input_data = np.random.random((16, 224, 224, 3))
        
        with pytest.raises(ValueError, match="No model loaded"):
            classifier.predict(input_data)
    
    @patch('gait_analysis.classification.gait_classifier.create_hybrid_cnn_lstm')
    def test_predict_basic(self, mock_create_hybrid):
        """Test basic prediction functionality."""
        # Mock the model and prediction
        mock_wrapper = Mock()
        mock_model = Mock()
        mock_wrapper.model = mock_model
        mock_model.count_params.return_value = 1000000
        mock_model.input_shape = (None, 16, 224, 224, 3)
        
        # Mock prediction output
        mock_predictions = np.array([[0.1, 0.8, 0.05, 0.03, 0.02]])
        mock_model.predict.return_value = mock_predictions
        
        mock_create_hybrid.return_value = mock_wrapper
        
        classifier = GaitClassifier(architecture_type='hybrid')
        classifier.build_model()
        
        # Test prediction
        input_data = np.random.random((16, 224, 224, 3))
        result = classifier.predict(input_data)
        
        assert isinstance(result, ClassificationResult)
        assert result.abnormality_type == 'limping'  # Index 1 has highest probability
        assert result.confidence == 0.8
        assert result.severity_score > 0
        assert isinstance(result.affected_limbs, list)
    
    @patch('gait_analysis.classification.gait_classifier.create_hybrid_cnn_lstm')
    def test_predict_with_probabilities(self, mock_create_hybrid):
        """Test prediction with probability output."""
        # Mock the model and prediction
        mock_wrapper = Mock()
        mock_model = Mock()
        mock_wrapper.model = mock_model
        mock_model.count_params.return_value = 1000000
        mock_model.input_shape = (None, 16, 224, 224, 3)
        
        # Mock prediction output
        mock_predictions = np.array([[0.6, 0.2, 0.1, 0.05, 0.05]])
        mock_model.predict.return_value = mock_predictions
        
        mock_create_hybrid.return_value = mock_wrapper
        
        classifier = GaitClassifier(architecture_type='hybrid')
        classifier.build_model()
        
        # Test prediction with probabilities
        input_data = np.random.random((16, 224, 224, 3))
        result = classifier.predict(input_data, return_probabilities=True)
        
        assert isinstance(result, dict)
        assert 'classification_result' in result
        assert 'class_probabilities' in result
        assert 'inference_time_ms' in result
        
        # Check classification result
        classification = result['classification_result']
        assert isinstance(classification, ClassificationResult)
        assert classification.abnormality_type == 'normal'  # Index 0 has highest probability
        
        # Check probabilities
        probabilities = result['class_probabilities']
        assert len(probabilities) == len(classifier.class_labels)
        assert all(0 <= prob <= 1 for prob in probabilities.values())
        assert abs(sum(probabilities.values()) - 1.0) < 1e-6  # Should sum to 1
    
    @patch('gait_analysis.classification.gait_classifier.create_hybrid_cnn_lstm')
    def test_predict_with_uncertainty(self, mock_create_hybrid):
        """Test prediction with uncertainty metrics."""
        # Mock the model and prediction
        mock_wrapper = Mock()
        mock_model = Mock()
        mock_wrapper.model = mock_model
        mock_model.count_params.return_value = 1000000
        mock_model.input_shape = (None, 16, 224, 224, 3)
        
        # Mock prediction output with high uncertainty
        mock_predictions = np.array([[0.3, 0.25, 0.2, 0.15, 0.1]])
        mock_model.predict.return_value = mock_predictions
        
        mock_create_hybrid.return_value = mock_wrapper
        
        classifier = GaitClassifier(architecture_type='hybrid')
        classifier.build_model()
        
        # Test prediction with uncertainty
        input_data = np.random.random((16, 224, 224, 3))
        result = classifier.predict(input_data, return_uncertainty=True)
        
        assert isinstance(result, dict)
        assert 'uncertainty_metrics' in result
        
        # Check uncertainty metrics
        uncertainty = result['uncertainty_metrics']
        assert 'entropy' in uncertainty
        assert 'max_probability' in uncertainty
        assert 'margin' in uncertainty
        assert 'gini_coefficient' in uncertainty
        assert 'uncertainty_score' in uncertainty
        
        # High entropy indicates high uncertainty
        assert uncertainty['entropy'] > 1.0
        assert 0 <= uncertainty['uncertainty_score'] <= 1
    
    def test_predict_proba(self):
        """Test probability prediction method."""
        classifier = GaitClassifier()
        
        # Mock the predict method to return probabilities
        mock_result = {
            'class_probabilities': {
                'normal': 0.7,
                'limping': 0.2,
                'shuffling': 0.05,
                'irregular_stride': 0.03,
                'balance_issues': 0.02
            }
        }
        
        with patch.object(classifier, 'predict', return_value=mock_result):
            input_data = np.random.random((16, 224, 224, 3))
            probabilities = classifier.predict_proba(input_data)
            
            assert isinstance(probabilities, dict)
            assert len(probabilities) == len(classifier.class_labels)
            assert all(label in probabilities for label in classifier.class_labels)
            assert abs(sum(probabilities.values()) - 1.0) < 1e-6
    
    def test_is_prediction_reliable_high_confidence(self):
        """Test reliability check with high confidence prediction."""
        classifier = GaitClassifier(confidence_threshold=0.7)
        
        # High confidence normal gait
        result = ClassificationResult(
            abnormality_type='normal',
            confidence=0.9,
            severity_score=0.1,
            affected_limbs=[]
        )
        
        assert classifier.is_prediction_reliable(result) is True
    
    def test_is_prediction_reliable_low_confidence(self):
        """Test reliability check with low confidence prediction."""
        classifier = GaitClassifier(confidence_threshold=0.7)
        
        # Low confidence prediction
        result = ClassificationResult(
            abnormality_type='limping',
            confidence=0.5,
            severity_score=0.6,
            affected_limbs=['left_leg']
        )
        
        assert classifier.is_prediction_reliable(result) is False
    
    def test_is_prediction_reliable_with_uncertainty(self):
        """Test reliability check with uncertainty metrics."""
        classifier = GaitClassifier(
            confidence_threshold=0.7,
            uncertainty_threshold=0.3
        )
        
        # High confidence but high uncertainty
        result = ClassificationResult(
            abnormality_type='limping',
            confidence=0.8,
            severity_score=0.7,
            affected_limbs=['left_leg']
        )
        
        uncertainty_metrics = {'entropy': 0.5}  # High uncertainty
        
        assert classifier.is_prediction_reliable(result, uncertainty_metrics) is False
    
    def test_get_performance_metrics_no_predictions(self):
        """Test performance metrics with no predictions made."""
        classifier = GaitClassifier()
        metrics = classifier.get_performance_metrics()
        
        assert metrics['total_predictions'] == 0
        assert metrics['avg_inference_time_ms'] == 0
        assert metrics['fps_capability'] == 0
    
    def test_get_performance_metrics_with_predictions(self):
        """Test performance metrics after making predictions."""
        classifier = GaitClassifier()
        
        # Simulate some inference times
        classifier.inference_times = [0.1, 0.15, 0.12, 0.08, 0.11]
        
        metrics = classifier.get_performance_metrics()
        
        assert metrics['total_predictions'] == 5
        assert metrics['avg_inference_time_ms'] > 0
        assert metrics['min_inference_time_ms'] > 0
        assert metrics['max_inference_time_ms'] > 0
        assert metrics['fps_capability'] > 0
        assert 'architecture_type' in metrics
    
    def test_reset_performance_tracking(self):
        """Test resetting performance tracking."""
        classifier = GaitClassifier()
        
        # Add some data
        classifier.inference_times = [0.1, 0.2]
        classifier.prediction_history = [{'test': 'data'}]
        
        classifier.reset_performance_tracking()
        
        assert len(classifier.inference_times) == 0
        assert len(classifier.prediction_history) == 0
    
    def test_get_expected_input_shape_3dcnn(self):
        """Test expected input shape for 3D-CNN architecture."""
        classifier = GaitClassifier(architecture_type='3dcnn')
        expected_shape = classifier._get_expected_input_shape()
        
        assert expected_shape == (16, 224, 224, 3)
    
    def test_get_expected_input_shape_lstm(self):
        """Test expected input shape for LSTM architecture."""
        classifier = GaitClassifier(architecture_type='lstm')
        expected_shape = classifier._get_expected_input_shape()
        
        assert expected_shape == (30, 66)
    
    def test_get_expected_input_shape_hybrid(self):
        """Test expected input shape for hybrid architecture."""
        classifier = GaitClassifier(architecture_type='hybrid')
        expected_shape = classifier._get_expected_input_shape()
        
        assert expected_shape == (16, 224, 224, 3)
    
    def test_validate_input_shape_valid(self):
        """Test input shape validation with valid shapes."""
        classifier = GaitClassifier(architecture_type='hybrid')
        
        # Valid shape without batch dimension
        assert classifier._validate_input_shape((16, 224, 224, 3), (16, 224, 224, 3)) is True
        
        # Valid shape with batch dimension
        assert classifier._validate_input_shape((1, 16, 224, 224, 3), (16, 224, 224, 3)) is True
    
    def test_validate_input_shape_invalid(self):
        """Test input shape validation with invalid shapes."""
        classifier = GaitClassifier(architecture_type='hybrid')
        
        # Wrong dimensions
        assert classifier._validate_input_shape((32, 224, 224, 3), (16, 224, 224, 3)) is False
        
        # Wrong number of dimensions
        assert classifier._validate_input_shape((224, 224, 3), (16, 224, 224, 3)) is False
    
    def test_validate_input_shape_lstm_variable_length(self):
        """Test LSTM input shape validation with variable sequence length."""
        classifier = GaitClassifier(architecture_type='lstm')
        
        # Variable sequence length should be allowed
        assert classifier._validate_input_shape((50, 66), (30, 66)) is True
        assert classifier._validate_input_shape((10, 66), (30, 66)) is True
        
        # Wrong feature dimension should fail
        assert classifier._validate_input_shape((30, 64), (30, 66)) is False
    
    def test_calculate_uncertainty_metrics(self):
        """Test uncertainty calculation from probabilities."""
        classifier = GaitClassifier()
        
        # High confidence (low uncertainty)
        high_conf_probs = np.array([0.9, 0.05, 0.02, 0.02, 0.01])
        uncertainty_low = classifier._calculate_uncertainty(high_conf_probs)
        
        # Low confidence (high uncertainty)
        low_conf_probs = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        uncertainty_high = classifier._calculate_uncertainty(low_conf_probs)
        
        # High confidence should have lower entropy
        assert uncertainty_low['entropy'] < uncertainty_high['entropy']
        assert uncertainty_low['max_probability'] > uncertainty_high['max_probability']
        assert uncertainty_low['uncertainty_score'] < uncertainty_high['uncertainty_score']
        
        # Check all metrics are present
        for uncertainty in [uncertainty_low, uncertainty_high]:
            assert 'entropy' in uncertainty
            assert 'max_probability' in uncertainty
            assert 'margin' in uncertainty
            assert 'gini_coefficient' in uncertainty
            assert 'uncertainty_score' in uncertainty
    
    def test_calculate_severity_score(self):
        """Test severity score calculation."""
        classifier = GaitClassifier()
        
        # Normal gait (low severity)
        normal_probs = np.array([0.9, 0.05, 0.02, 0.02, 0.01])
        severity_normal = classifier._calculate_severity_score(normal_probs)
        
        # Abnormal gait (high severity)
        abnormal_probs = np.array([0.1, 0.8, 0.05, 0.03, 0.02])
        severity_abnormal = classifier._calculate_severity_score(abnormal_probs)
        
        # Abnormal should have higher severity
        assert severity_abnormal > severity_normal
        assert 0 <= severity_normal <= 1
        assert 0 <= severity_abnormal <= 1
    
    def test_determine_affected_limbs(self):
        """Test affected limbs determination."""
        classifier = GaitClassifier()
        
        # Normal gait should have no affected limbs
        affected_normal = classifier._determine_affected_limbs(0, 0.9)
        assert affected_normal == []
        
        # Low confidence should have no affected limbs
        affected_low_conf = classifier._determine_affected_limbs(1, 0.3)
        assert affected_low_conf == []
        
        # Limping should affect legs
        affected_limping = classifier._determine_affected_limbs(1, 0.8)  # limping
        assert 'left_leg' in affected_limping or 'right_leg' in affected_limping
    
    def test_save_and_load_model_metadata(self):
        """Test saving and loading model with metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.h5"
            metadata_path = Path(temp_dir) / "test_model_metadata.json"
            
            classifier = GaitClassifier(architecture_type='hybrid')
            
            # Mock model wrapper for saving
            mock_wrapper = Mock()
            classifier.model_wrapper = mock_wrapper
            classifier.model = Mock()
            classifier.model.count_params.return_value = 1000000
            
            # Save model
            classifier.save_model(str(model_path))
            
            # Check metadata was created
            assert metadata_path.exists()
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            assert metadata['architecture_type'] == 'hybrid'
            assert metadata['class_labels'] == classifier.class_labels
            assert metadata['num_classes'] == classifier.num_classes
    
    def test_string_representations(self):
        """Test string and repr methods."""
        classifier = GaitClassifier(architecture_type='3dcnn')
        
        # Test __str__
        str_repr = str(classifier)
        assert 'GaitClassifier' in str_repr
        assert '3dcnn' in str_repr
        assert 'not loaded' in str_repr
        
        # Test __repr__
        repr_str = repr(classifier)
        assert 'GaitClassifier' in repr_str
        assert 'architecture_type' in repr_str
        assert 'model_loaded=False' in repr_str


if __name__ == '__main__':
    pytest.main([__file__])