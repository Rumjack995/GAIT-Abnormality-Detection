"""
Demo script for the three model architectures optimized for RTX 4050.

This script demonstrates how to create, compile, and use the three different
model architectures for gait abnormality detection:
1. Lightweight 3D-CNN
2. Efficient LSTM with attention
3. Hybrid CNN-LSTM

Run this script to see the models in action and compare their characteristics.
"""

import numpy as np
import tensorflow as tf
from gait_analysis.models import (
    create_lightweight_3dcnn,
    create_efficient_lstm,
    create_hybrid_cnn_lstm
)


def demonstrate_3dcnn():
    """Demonstrate the Lightweight 3D-CNN architecture."""
    print("=" * 60)
    print("LIGHTWEIGHT 3D-CNN ARCHITECTURE")
    print("=" * 60)
    
    # Create model with realistic parameters for RTX 4050
    model = create_lightweight_3dcnn(
        input_shape=(16, 224, 224, 3),  # 16 frames, 224x224 resolution
        num_classes=5,  # 5 gait abnormality classes
        use_mixed_precision=False  # Disable for demo
    )
    
    print(f"Model created successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Number of classes: {model.num_classes}")
    print(f"Total parameters: {model.model.count_params():,}")
    
    # Create dummy video data
    dummy_video = np.random.random((2, 16, 224, 224, 3)).astype(np.float32)
    print(f"Dummy input shape: {dummy_video.shape}")
    
    # Make predictions
    predictions = model.model.predict(dummy_video, verbose=0)
    print(f"Prediction shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[0]}")
    
    print("\n3D-CNN Model Summary:")
    print(model.get_model_summary())


def demonstrate_lstm():
    """Demonstrate the Efficient LSTM architecture."""
    print("=" * 60)
    print("EFFICIENT LSTM ARCHITECTURE")
    print("=" * 60)
    
    # Create model optimized for pose sequences
    model = create_efficient_lstm(
        input_shape=(30, 66),  # 30 frames, 66 features (33 keypoints * 2D)
        num_classes=5,
        lstm_units=64,
        attention_dim=64,
        use_mixed_precision=False
    )
    
    print(f"Model created successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"LSTM units: {model.lstm_units}")
    print(f"Attention dimension: {model.attention_dim}")
    print(f"Total parameters: {model.model.count_params():,}")
    
    # Create dummy pose sequence data
    dummy_poses = np.random.random((4, 30, 66)).astype(np.float32)
    print(f"Dummy input shape: {dummy_poses.shape}")
    
    # Make predictions
    predictions = model.model.predict(dummy_poses, verbose=0)
    print(f"Prediction shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[0]}")
    
    # Test attention mechanism
    predictions_with_attention, attention_weights = model.predict_with_attention(dummy_poses)
    print(f"Predictions with attention: {predictions_with_attention.shape}")
    if attention_weights is not None:
        print(f"Attention weights shape: {attention_weights.shape}")
    
    print("\nLSTM Model Summary:")
    print(model.get_model_summary())


def demonstrate_hybrid():
    """Demonstrate the Hybrid CNN-LSTM architecture."""
    print("=" * 60)
    print("HYBRID CNN-LSTM ARCHITECTURE")
    print("=" * 60)
    
    # Create hybrid model combining CNN and LSTM
    model = create_hybrid_cnn_lstm(
        frame_shape=(224, 224, 3),
        sequence_length=16,
        num_classes=5,
        spatial_feature_dim=128,
        lstm_units=64,
        fusion_dim=128,
        trainable_backbone=False,  # Keep MobileNetV2 frozen for demo
        use_mixed_precision=False
    )
    
    print(f"Model created successfully!")
    print(f"Frame shape: {model.frame_shape}")
    print(f"Sequence length: {model.sequence_length}")
    print(f"Spatial feature dim: {model.spatial_feature_dim}")
    print(f"LSTM units: {model.lstm_units}")
    print(f"Fusion dim: {model.fusion_dim}")
    print(f"Total parameters: {model.model.count_params():,}")
    
    # Create dummy video sequence
    dummy_video = np.random.random((2, 16, 224, 224, 3)).astype(np.float32)
    print(f"Dummy input shape: {dummy_video.shape}")
    
    # Make predictions
    predictions = model.model.predict(dummy_video, verbose=0)
    print(f"Prediction shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[0]}")
    
    # Test real-time prediction
    single_video = dummy_video[0]  # Take first sample
    realtime_result = model.predict_realtime(single_video)
    print(f"\nReal-time prediction result:")
    print(f"  Predicted class: {realtime_result['predicted_class']}")
    print(f"  Confidence: {realtime_result['confidence']:.4f}")
    print(f"  Inference time: {realtime_result['inference_time_ms']:.2f} ms")
    print(f"  FPS capability: {realtime_result['fps_capability']:.1f}")
    
    print("\nHybrid Model Summary:")
    print(model.get_model_summary())


def compare_architectures():
    """Compare the three architectures side by side."""
    print("=" * 60)
    print("ARCHITECTURE COMPARISON")
    print("=" * 60)
    
    # Create smaller models for fair comparison
    models = {
        "3D-CNN": create_lightweight_3dcnn(
            input_shape=(8, 112, 112, 3),
            num_classes=5,
            use_mixed_precision=False
        ),
        "LSTM": create_efficient_lstm(
            input_shape=(30, 66),
            num_classes=5,
            use_mixed_precision=False
        ),
        "Hybrid": create_hybrid_cnn_lstm(
            frame_shape=(112, 112, 3),
            sequence_length=8,
            num_classes=5,
            use_mixed_precision=False
        )
    }
    
    print(f"{'Architecture':<15} {'Parameters':<15} {'Memory (MB)':<15} {'Input Type':<20}")
    print("-" * 70)
    
    for name, model in models.items():
        param_count = model.model.count_params()
        # Rough memory estimate (parameters * 4 bytes)
        memory_mb = (param_count * 4) / (1024 * 1024)
        
        if name == "3D-CNN":
            input_type = "Video frames"
        elif name == "LSTM":
            input_type = "Pose sequences"
        else:
            input_type = "Video sequences"
        
        print(f"{name:<15} {param_count:<15,} {memory_mb:<15.1f} {input_type:<20}")
    
    print("\nKey Characteristics:")
    print("- 3D-CNN: Best for spatiotemporal pattern recognition")
    print("- LSTM: Best for long-term temporal dependencies")
    print("- Hybrid: Best balance of spatial and temporal features")
    print("- All models optimized for RTX 4050 (6GB VRAM)")


def main():
    """Run all demonstrations."""
    print("GAIT ABNORMALITY DETECTION - MODEL ARCHITECTURES DEMO")
    print("Optimized for RTX 4050 (6GB VRAM)")
    print("=" * 60)
    
    try:
        # Demonstrate each architecture
        demonstrate_3dcnn()
        print("\n")
        
        demonstrate_lstm()
        print("\n")
        
        demonstrate_hybrid()
        print("\n")
        
        # Compare architectures
        compare_architectures()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("All three model architectures are ready for training.")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set TensorFlow to use CPU for demo (avoid GPU memory issues)
    tf.config.set_visible_devices([], 'GPU')
    
    main()