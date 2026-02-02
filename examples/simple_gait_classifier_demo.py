"""
Simple Gait Classifier Demonstration

This script demonstrates the basic usage of the GaitClassifier class
with CPU-compatible settings.
"""

import numpy as np
import time
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gait_analysis.classification.gait_classifier import GaitClassifier
from gait_analysis.utils.data_structures import ClassificationResult


def simple_demo():
    """Simple demonstration of GaitClassifier functionality."""
    
    print("=== Simple Gait Classifier Demo ===\n")
    
    # Test LSTM architecture (most CPU-friendly)
    print("--- Testing LSTM Architecture ---")
    
    try:
        # Create classifier with explicit parameters
        classifier = GaitClassifier(
            architecture_type='lstm',
            confidence_threshold=0.7,
            uncertainty_threshold=0.3
        )
        
        print(f"Created classifier: {classifier}")
        
        # Build model with explicit parameters (CPU-friendly)
        print("Building LSTM model...")
        classifier.build_model(
            input_shape=(30, 66),  # 30 frames, 66 features (33 keypoints * 2D)
            lstm_units=32,  # Smaller for CPU
            use_mixed_precision=False  # Disable for CPU compatibility
        )
        
        print(f"Model built successfully with {classifier.model.count_params():,} parameters")
        
        # Generate sample pose sequence input
        input_data = np.random.random((30, 66))
        print(f"Generated pose sequence input: {input_data.shape}")
        
        # Make basic prediction
        print("\nMaking basic prediction...")
        start_time = time.time()
        result = classifier.predict(input_data)
        prediction_time = time.time() - start_time
        
        print(f"Prediction completed in {prediction_time*1000:.2f}ms")
        print(f"Predicted abnormality: {result.abnormality_type}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Severity score: {result.severity_score:.3f}")
        print(f"Affected limbs: {result.affected_limbs}")
        
        # Check prediction reliability
        is_reliable = classifier.is_prediction_reliable(result)
        print(f"Prediction reliable: {is_reliable}")
        
        # Make prediction with probabilities
        print("\nMaking prediction with probabilities...")
        detailed_result = classifier.predict(
            input_data, 
            return_probabilities=True
        )
        
        print("Class probabilities:")
        for class_name, prob in detailed_result['class_probabilities'].items():
            print(f"  {class_name}: {prob:.3f}")
        
        # Test predict_proba method
        print("\nTesting predict_proba method...")
        probabilities = classifier.predict_proba(input_data)
        print("Probabilities from predict_proba:")
        for class_name, prob in probabilities.items():
            print(f"  {class_name}: {prob:.3f}")
        
        # Get performance metrics
        print("\nPerformance metrics:")
        metrics = classifier.get_performance_metrics()
        print(f"  Average inference time: {metrics['avg_inference_time_ms']:.2f}ms")
        print(f"  FPS capability: {metrics['fps_capability']:.1f}")
        print(f"  Total predictions: {metrics['total_predictions']}")
        
        print("\nLSTM architecture test completed successfully!")
        
    except Exception as e:
        print(f"Error testing LSTM architecture: {str(e)}")
        import traceback
        traceback.print_exc()


def uncertainty_demo():
    """Demonstrate uncertainty analysis."""
    
    print("\n=== Uncertainty Analysis Demo ===\n")
    
    try:
        classifier = GaitClassifier(architecture_type='lstm')
        classifier.build_model(
            input_shape=(30, 66),
            lstm_units=32,
            use_mixed_precision=False
        )
        
        # Test different confidence scenarios
        scenarios = [
            ("High Confidence Normal", np.array([0.95, 0.02, 0.01, 0.01, 0.01])),
            ("High Confidence Abnormal", np.array([0.05, 0.90, 0.02, 0.02, 0.01])),
            ("Low Confidence/Uncertain", np.array([0.3, 0.25, 0.2, 0.15, 0.1])),
            ("Moderate Confidence", np.array([0.1, 0.6, 0.15, 0.1, 0.05]))
        ]
        
        for scenario_name, mock_probs in scenarios:
            print(f"--- {scenario_name} ---")
            
            # Calculate uncertainty metrics
            uncertainty = classifier._calculate_uncertainty(mock_probs)
            
            print(f"Probabilities: {mock_probs}")
            print(f"Entropy: {uncertainty['entropy']:.3f}")
            print(f"Max probability: {uncertainty['max_probability']:.3f}")
            print(f"Margin: {uncertainty['margin']:.3f}")
            print(f"Uncertainty score: {uncertainty['uncertainty_score']:.3f}")
            
            # Create mock classification result
            predicted_class = np.argmax(mock_probs)
            result = ClassificationResult(
                abnormality_type=classifier.class_labels[predicted_class],
                confidence=float(mock_probs[predicted_class]),
                severity_score=classifier._calculate_severity_score(mock_probs),
                affected_limbs=classifier._determine_affected_limbs(predicted_class, float(mock_probs[predicted_class]))
            )
            
            # Check reliability
            is_reliable = classifier.is_prediction_reliable(result, uncertainty)
            print(f"Prediction reliable: {is_reliable}")
            print(f"Severity score: {result.severity_score:.3f}")
            print()
            
    except Exception as e:
        print(f"Error in uncertainty demo: {str(e)}")
        import traceback
        traceback.print_exc()


def performance_demo():
    """Demonstrate performance tracking."""
    
    print("\n=== Performance Tracking Demo ===\n")
    
    try:
        classifier = GaitClassifier(architecture_type='lstm')
        classifier.build_model(
            input_shape=(30, 66),
            lstm_units=32,
            use_mixed_precision=False
        )
        
        print("Running multiple predictions to track performance...")
        
        # Run multiple predictions
        for i in range(5):
            input_data = np.random.random((30, 66))
            result = classifier.predict(input_data)
            print(f"Prediction {i+1}: {result.abnormality_type} (confidence: {result.confidence:.3f})")
        
        # Get final performance metrics
        print("\nFinal performance metrics:")
        metrics = classifier.get_performance_metrics()
        print(f"  Total predictions: {metrics['total_predictions']}")
        print(f"  Average inference time: {metrics['avg_inference_time_ms']:.2f}ms")
        print(f"  Min inference time: {metrics['min_inference_time_ms']:.2f}ms")
        print(f"  Max inference time: {metrics['max_inference_time_ms']:.2f}ms")
        print(f"  FPS capability: {metrics['fps_capability']:.1f}")
        
        # Reset and verify
        classifier.reset_performance_tracking()
        metrics_after_reset = classifier.get_performance_metrics()
        print(f"\nAfter reset - Total predictions: {metrics_after_reset['total_predictions']}")
        
    except Exception as e:
        print(f"Error in performance demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    simple_demo()
    uncertainty_demo()
    performance_demo()
    print("\n=== All Demos Complete ===")