"""
Gait Classifier Demonstration

This script demonstrates the usage of the GaitClassifier class with different
architectures for gait abnormality detection and classification.
"""

import numpy as np
import time
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gait_analysis.classification.gait_classifier import GaitClassifier
from gait_analysis.utils.data_structures import ClassificationResult


def demonstrate_gait_classifier():
    """Demonstrate GaitClassifier functionality with different architectures."""
    
    print("=== Gait Classifier Demonstration ===\n")
    
    # Test different architectures
    architectures = ['3dcnn', 'lstm', 'hybrid']
    
    for arch in architectures:
        print(f"--- Testing {arch.upper()} Architecture ---")
        
        try:
            # Create classifier
            classifier = GaitClassifier(
                architecture_type=arch,
                confidence_threshold=0.7,
                uncertainty_threshold=0.3
            )
            
            print(f"Created classifier: {classifier}")
            
            # Build model
            print("Building model...")
            classifier.build_model()
            print(f"Model built successfully with {classifier.model.count_params():,} parameters")
            
            # Generate sample input data based on architecture
            if arch in ['3dcnn', 'hybrid']:
                # Video sequence input (frames, height, width, channels)
                input_data = np.random.random((16, 224, 224, 3))
                print(f"Generated video input: {input_data.shape}")
            else:  # lstm
                # Pose sequence input (sequence_length, features)
                input_data = np.random.random((30, 66))  # 33 keypoints * 2D
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
            
            # Make prediction with probabilities and uncertainty
            print("\nMaking detailed prediction...")
            detailed_result = classifier.predict(
                input_data, 
                return_probabilities=True, 
                return_uncertainty=True
            )
            
            print("Class probabilities:")
            for class_name, prob in detailed_result['class_probabilities'].items():
                print(f"  {class_name}: {prob:.3f}")
            
            print("Uncertainty metrics:")
            uncertainty = detailed_result['uncertainty_metrics']
            print(f"  Entropy: {uncertainty['entropy']:.3f}")
            print(f"  Max probability: {uncertainty['max_probability']:.3f}")
            print(f"  Uncertainty score: {uncertainty['uncertainty_score']:.3f}")
            
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
            
            print(f"\n{arch.upper()} architecture test completed successfully!\n")
            
        except Exception as e:
            print(f"Error testing {arch} architecture: {str(e)}\n")
    
    # Demonstrate model comparison
    print("--- Model Architecture Comparison ---")
    
    comparison_results = []
    
    for arch in architectures:
        try:
            classifier = GaitClassifier(architecture_type=arch)
            classifier.build_model()
            
            # Generate appropriate input
            if arch in ['3dcnn', 'hybrid']:
                test_input = np.random.random((1, 16, 224, 224, 3))
            else:
                test_input = np.random.random((1, 30, 66))
            
            # Measure inference time
            start_time = time.time()
            for _ in range(10):  # Average over 10 predictions
                _ = classifier.predict(test_input)
            avg_time = (time.time() - start_time) / 10
            
            comparison_results.append({
                'architecture': arch,
                'parameters': classifier.model.count_params(),
                'avg_inference_time_ms': avg_time * 1000,
                'fps_capability': 1.0 / avg_time
            })
            
        except Exception as e:
            print(f"Error in comparison for {arch}: {str(e)}")
    
    # Display comparison results
    print("\nArchitecture Comparison Results:")
    print(f"{'Architecture':<12} {'Parameters':<12} {'Inference (ms)':<15} {'FPS':<8}")
    print("-" * 50)
    
    for result in comparison_results:
        print(f"{result['architecture']:<12} "
              f"{result['parameters']:,<12} "
              f"{result['avg_inference_time_ms']:<15.2f} "
              f"{result['fps_capability']:<8.1f}")
    
    print("\n=== Demonstration Complete ===")


def demonstrate_uncertainty_analysis():
    """Demonstrate uncertainty analysis capabilities."""
    
    print("\n=== Uncertainty Analysis Demonstration ===\n")
    
    classifier = GaitClassifier(architecture_type='hybrid')
    classifier.build_model()
    
    # Generate different types of predictions to show uncertainty
    scenarios = [
        ("High Confidence Normal", np.array([[0.95, 0.02, 0.01, 0.01, 0.01]])),
        ("High Confidence Abnormal", np.array([[0.05, 0.90, 0.02, 0.02, 0.01]])),
        ("Low Confidence/Uncertain", np.array([[0.3, 0.25, 0.2, 0.15, 0.1]])),
        ("Moderate Confidence", np.array([[0.1, 0.6, 0.15, 0.1, 0.05]]))
    ]
    
    for scenario_name, mock_probs in scenarios:
        print(f"--- {scenario_name} ---")
        
        # Calculate uncertainty metrics
        uncertainty = classifier._calculate_uncertainty(mock_probs[0])
        
        print(f"Probabilities: {mock_probs[0]}")
        print(f"Entropy: {uncertainty['entropy']:.3f}")
        print(f"Max probability: {uncertainty['max_probability']:.3f}")
        print(f"Margin: {uncertainty['margin']:.3f}")
        print(f"Uncertainty score: {uncertainty['uncertainty_score']:.3f}")
        
        # Create mock classification result
        predicted_class = np.argmax(mock_probs[0])
        result = ClassificationResult(
            abnormality_type=classifier.class_labels[predicted_class],
            confidence=float(mock_probs[0][predicted_class]),
            severity_score=classifier._calculate_severity_score(mock_probs[0]),
            affected_limbs=classifier._determine_affected_limbs(predicted_class, float(mock_probs[0][predicted_class]))
        )
        
        # Check reliability
        is_reliable = classifier.is_prediction_reliable(result, uncertainty)
        print(f"Prediction reliable: {is_reliable}")
        print(f"Severity score: {result.severity_score:.3f}")
        print()


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_gait_classifier()
    demonstrate_uncertainty_analysis()