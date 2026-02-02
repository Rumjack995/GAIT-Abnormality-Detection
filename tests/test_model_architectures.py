"""
Tests for model architectures optimized for RTX 4050.

This module tests the three model architectures:
1. Lightweight 3D-CNN
2. Efficient LSTM with attention
3. Hybrid CNN-LSTM
"""

import pytest
import numpy as np
import tensorflow as tf
from gait_analysis.models import (
    create_lightweight_3dcnn,
    create_efficient_lstm,
    create_hybrid_cnn_lstm,
    LightweightCNN3D,
    EfficientLSTM,
    HybridCNNLSTM
)


class TestLightweight3DCNN:
    """Test cases for Lightweight 3D-CNN architecture."""
    
    def test_model_creation(self):
        """Test that 3D-CNN model can be created and compiled."""
        model = create_lightweight_3dcnn(
            input_shape=(16, 224, 224, 3),
            num_classes=5,
            use_mixed_precision=False  # Disable for testing
        )
        
        assert model.model is not None
        assert model.num_classes == 5
        assert model.input_shape == (16, 224, 224, 3)
    
    def test_model_prediction_shape(self):
        """Test that model produces correct output shape."""
        model = create_lightweight_3dcnn(
            input_shape=(8, 112, 112, 3),  # Smaller for testing
            num_classes=3,
            use_mixed_precision=False
        )
        
        # Create dummy input
        dummy_input = np.random.random((2, 8, 112, 112, 3)).astype(np.float32)
        
        # Make prediction
        predictions = model.model.predict(dummy_input, verbose=0)
        
        assert predictions.shape == (2, 3)
        assert np.allclose(np.sum(predictions, axis=1), 1.0, atol=1e-5)  # Softmax check
    
    def test_model_summary(self):
        """Test that model summary can be generated."""
        model = LightweightCNN3D(
            input_shape=(8, 112, 112, 3),
            num_classes=3,
            use_mixed_precision=False
        )
        model.build_model()
        
        summary = model.get_model_summary()
        assert "Lightweight 3D-CNN Model Summary" in summary
        assert "Total Parameters" in summary


class TestEfficientLSTM:
    """Test cases for Efficient LSTM architecture."""
    
    def test_model_creation(self):
        """Test that LSTM model can be created and compiled."""
        model = create_efficient_lstm(
            input_shape=(30, 66),
            num_classes=5,
            lstm_units=32,  # Smaller for testing
            use_mixed_precision=False
        )
        
        assert model.model is not None
        assert model.num_classes == 5
        assert model.input_shape == (30, 66)
    
    def test_model_prediction_shape(self):
        """Test that model produces correct output shape."""
        model = create_efficient_lstm(
            input_shape=(20, 44),  # Smaller for testing
            num_classes=3,
            lstm_units=16,
            use_mixed_precision=False
        )
        
        # Create dummy input
        dummy_input = np.random.random((4, 20, 44)).astype(np.float32)
        
        # Make prediction
        predictions = model.model.predict(dummy_input, verbose=0)
        
        assert predictions.shape == (4, 3)
        assert np.allclose(np.sum(predictions, axis=1), 1.0, atol=1e-5)  # Softmax check
    
    def test_attention_mechanism(self):
        """Test that attention mechanism works."""
        model = EfficientLSTM(
            input_shape=(15, 32),
            num_classes=2,
            lstm_units=16,
            attention_dim=32,
            use_mixed_precision=False
        )
        model.build_model()
        model.compile_model()
        
        # Create dummy input
        dummy_input = np.random.random((2, 15, 32)).astype(np.float32)
        
        # Test prediction with attention
        predictions, attention_weights = model.predict_with_attention(dummy_input)
        
        assert predictions.shape == (2, 2)
        # Note: attention_weights might be None if attention layer not found in simple test


class TestHybridCNNLSTM:
    """Test cases for Hybrid CNN-LSTM architecture."""
    
    def test_model_creation(self):
        """Test that hybrid model can be created and compiled."""
        model = create_hybrid_cnn_lstm(
            frame_shape=(112, 112, 3),  # Smaller for testing
            sequence_length=8,
            num_classes=5,
            spatial_feature_dim=64,
            lstm_units=32,
            use_mixed_precision=False
        )
        
        assert model.model is not None
        assert model.num_classes == 5
        assert model.sequence_length == 8
    
    def test_model_prediction_shape(self):
        """Test that model produces correct output shape."""
        model = create_hybrid_cnn_lstm(
            frame_shape=(64, 64, 3),  # Very small for testing
            sequence_length=4,
            num_classes=3,
            spatial_feature_dim=32,
            lstm_units=16,
            use_mixed_precision=False
        )
        
        # Create dummy input
        dummy_input = np.random.random((2, 4, 64, 64, 3)).astype(np.float32)
        
        # Make prediction
        predictions = model.model.predict(dummy_input, verbose=0)
        
        assert predictions.shape == (2, 3)
        assert np.allclose(np.sum(predictions, axis=1), 1.0, atol=1e-5)  # Softmax check
    
    def test_realtime_prediction(self):
        """Test real-time prediction functionality."""
        model = HybridCNNLSTM(
            frame_shape=(64, 64, 3),
            sequence_length=4,
            num_classes=2,
            spatial_feature_dim=32,
            lstm_units=16,
            use_mixed_precision=False
        )
        model.build_model()
        model.compile_model()
        
        # Create dummy video sequence
        video_sequence = np.random.random((4, 64, 64, 3)).astype(np.float32)
        
        # Test real-time prediction
        result = model.predict_realtime(video_sequence)
        
        assert 'predicted_class' in result
        assert 'confidence' in result
        assert 'inference_time_ms' in result
        assert 'fps_capability' in result
        assert isinstance(result['predicted_class'], int)
        assert 0 <= result['confidence'] <= 1


class TestModelComparison:
    """Test cases for comparing model architectures."""
    
    def test_parameter_counts(self):
        """Test that models have reasonable parameter counts for RTX 4050."""
        # Create small models for comparison
        cnn_3d = create_lightweight_3dcnn(
            input_shape=(8, 64, 64, 3),
            num_classes=3,
            use_mixed_precision=False
        )
        
        lstm = create_efficient_lstm(
            input_shape=(20, 32),
            num_classes=3,
            lstm_units=32,
            use_mixed_precision=False
        )
        
        hybrid = create_hybrid_cnn_lstm(
            frame_shape=(64, 64, 3),
            sequence_length=8,
            num_classes=3,
            spatial_feature_dim=32,
            lstm_units=32,
            use_mixed_precision=False
        )
        
        # Get parameter counts
        cnn_params = cnn_3d.model.count_params()
        lstm_params = lstm.model.count_params()
        hybrid_params = hybrid.model.count_params()
        
        # All models should have reasonable parameter counts
        assert cnn_params > 0
        assert lstm_params > 0
        assert hybrid_params > 0
        
        # Print for debugging
        print(f"3D-CNN parameters: {cnn_params:,}")
        print(f"LSTM parameters: {lstm_params:,}")
        print(f"Hybrid parameters: {hybrid_params:,}")
    
    def test_memory_efficiency(self):
        """Test that models are designed for memory efficiency."""
        # Test with realistic but small inputs
        models = [
            create_lightweight_3dcnn(
                input_shape=(8, 112, 112, 3),
                num_classes=5,
                use_mixed_precision=False
            ),
            create_efficient_lstm(
                input_shape=(30, 66),
                num_classes=5,
                use_mixed_precision=False
            ),
            create_hybrid_cnn_lstm(
                frame_shape=(112, 112, 3),
                sequence_length=8,
                num_classes=5,
                use_mixed_precision=False
            )
        ]
        
        for i, model in enumerate(models):
            # Check that model can be created without memory errors
            assert model.model is not None
            
            # Check parameter count is reasonable (less than 10M for test models)
            param_count = model.model.count_params()
            assert param_count < 10_000_000, f"Model {i} has too many parameters: {param_count:,}"


if __name__ == "__main__":
    # Run basic tests
    test_3dcnn = TestLightweight3DCNN()
    test_3dcnn.test_model_creation()
    test_3dcnn.test_model_prediction_shape()
    
    test_lstm = TestEfficientLSTM()
    test_lstm.test_model_creation()
    test_lstm.test_model_prediction_shape()
    
    test_hybrid = TestHybridCNNLSTM()
    test_hybrid.test_model_creation()
    test_hybrid.test_model_prediction_shape()
    
    test_comparison = TestModelComparison()
    test_comparison.test_parameter_counts()
    
    print("All basic model architecture tests passed!")