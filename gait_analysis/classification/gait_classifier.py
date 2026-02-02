"""
Gait Classifier with multi-architecture support.

This module implements a unified interface for gait abnormality classification
supporting 3D-CNN, LSTM, and Hybrid CNN-LSTM architectures with model loading,
inference, confidence scoring, and uncertainty quantification.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
import json
import time
from dataclasses import asdict

from ..models import (
    create_lightweight_3dcnn, 
    create_efficient_lstm, 
    create_hybrid_cnn_lstm,
    LightweightCNN3D,
    EfficientLSTM, 
    HybridCNNLSTM
)
from ..utils.data_structures import (
    ClassificationResult, 
    PerformanceMetrics,
    ModelComparison
)
from ..utils.config import get_config


class GaitClassifier:
    """
    Unified gait classifier supporting multiple architectures.
    
    This class provides a consistent interface for gait abnormality detection
    and classification across different deep learning architectures:
    - 3D-CNN: For spatiotemporal pattern recognition
    - LSTM: For temporal sequence modeling  
    - Hybrid CNN-LSTM: For balanced spatial-temporal learning
    
    Features:
    - Multi-architecture support with automatic model selection
    - Confidence scoring and uncertainty quantification
    - Model loading and inference methods
    - Performance comparison and benchmarking
    - Real-time inference capabilities
    """
    
    # Supported architecture types
    SUPPORTED_ARCHITECTURES = ['3dcnn', 'lstm', 'hybrid']
    
    # Default class labels for gait abnormalities
    DEFAULT_CLASS_LABELS = [
        'normal',
        'limping', 
        'shuffling',
        'irregular_stride',
        'balance_issues'
    ]
    
    def __init__(self, 
                 architecture_type: str = 'hybrid',
                 class_labels: Optional[List[str]] = None,
                 confidence_threshold: float = 0.7,
                 uncertainty_threshold: float = 0.3,
                 model_path: Optional[str] = None):
        """
        Initialize the gait classifier.
        
        Args:
            architecture_type: Type of architecture ('3dcnn', 'lstm', 'hybrid')
            class_labels: List of class labels for classification
            confidence_threshold: Minimum confidence for reliable predictions
            uncertainty_threshold: Maximum uncertainty for reliable predictions
            model_path: Path to pre-trained model (optional)
            
        Raises:
            ValueError: If architecture_type is not supported
        """
        if architecture_type not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture: {architecture_type}. "
                f"Supported types: {self.SUPPORTED_ARCHITECTURES}"
            )
        
        self.architecture_type = architecture_type
        self.class_labels = class_labels or self.DEFAULT_CLASS_LABELS
        self.num_classes = len(self.class_labels)
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        
        # Model and configuration
        self.model = None
        self.model_wrapper = None
        self.config = get_config()
        self.logger = self._setup_logger()
        
        # Performance tracking
        self.inference_times = []
        self.prediction_history = []
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the classifier."""
        logger = logging.getLogger(f'GaitClassifier_{self.architecture_type}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def build_model(self, **kwargs) -> None:
        """
        Build the model architecture.
        
        Args:
            **kwargs: Architecture-specific parameters
        """
        self.logger.info(f"Building {self.architecture_type} model...")
        
        # Get default parameters from config
        model_config = self.config.model_config.get(self.architecture_type, {})
        
        # Filter parameters based on architecture type
        filtered_params = self._filter_model_params(model_config)
        
        # Merge with provided kwargs
        params = {**filtered_params, **kwargs}
        params['num_classes'] = self.num_classes
        
        # Disable mixed precision for CPU-only execution to avoid compatibility issues
        import tensorflow as tf
        if not tf.config.list_physical_devices('GPU'):
            params['use_mixed_precision'] = False
            self.logger.info("GPU not available, disabling mixed precision")
        
        try:
            if self.architecture_type == '3dcnn':
                self.model_wrapper = create_lightweight_3dcnn(**params)
                self.model = self.model_wrapper.model
                
            elif self.architecture_type == 'lstm':
                self.model_wrapper = create_efficient_lstm(**params)
                self.model = self.model_wrapper.model
                
            elif self.architecture_type == 'hybrid':
                self.model_wrapper = create_hybrid_cnn_lstm(**params)
                self.model = self.model_wrapper.model
            
            self.logger.info(f"Successfully built {self.architecture_type} model")
            self.logger.info(f"Model parameters: {self.model.count_params():,}")
            
        except Exception as e:
            self.logger.error(f"Failed to build model: {str(e)}")
            raise
    
    def load_model(self, model_path: str) -> None:
        """
        Load a pre-trained model from file.
        
        Args:
            model_path: Path to the saved model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model architecture doesn't match
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.logger.info(f"Loading model from {model_path}")
        
        try:
            # Load model metadata if available
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Validate architecture compatibility
                if metadata.get('architecture_type') != self.architecture_type:
                    self.logger.warning(
                        f"Model architecture mismatch: expected {self.architecture_type}, "
                        f"got {metadata.get('architecture_type')}"
                    )
                
                # Update class labels if provided in metadata
                if 'class_labels' in metadata:
                    self.class_labels = metadata['class_labels']
                    self.num_classes = len(self.class_labels)
            
            # Load the model based on architecture type
            if self.architecture_type == '3dcnn':
                self.model_wrapper = LightweightCNN3D(num_classes=self.num_classes)
                self.model_wrapper.load_model(str(model_path))
                self.model = self.model_wrapper.model
                
            elif self.architecture_type == 'lstm':
                self.model_wrapper = EfficientLSTM(num_classes=self.num_classes)
                self.model_wrapper.load_model(str(model_path))
                self.model = self.model_wrapper.model
                
            elif self.architecture_type == 'hybrid':
                self.model_wrapper = HybridCNNLSTM(num_classes=self.num_classes)
                self.model_wrapper.load_model(str(model_path))
                self.model = self.model_wrapper.model
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def save_model(self, model_path: str, include_metadata: bool = True) -> None:
        """
        Save the current model to file.
        
        Args:
            model_path: Path to save the model
            include_metadata: Whether to save model metadata
            
        Raises:
            ValueError: If no model is loaded
        """
        if self.model is None:
            raise ValueError("No model to save. Build or load a model first.")
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving model to {model_path}")
        
        try:
            # Save the model
            self.model_wrapper.save_model(str(model_path))
            
            # Save metadata
            if include_metadata:
                metadata = {
                    'architecture_type': self.architecture_type,
                    'class_labels': self.class_labels,
                    'num_classes': self.num_classes,
                    'confidence_threshold': self.confidence_threshold,
                    'uncertainty_threshold': self.uncertainty_threshold,
                    'model_parameters': self.model.count_params(),
                    'save_timestamp': time.time()
                }
                
                metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            self.logger.info("Model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def predict(self, 
                input_data: np.ndarray, 
                return_probabilities: bool = False,
                return_uncertainty: bool = False) -> Union[ClassificationResult, Dict[str, Any]]:
        """
        Make predictions on input data.
        
        Args:
            input_data: Input data for prediction
            return_probabilities: Whether to return class probabilities
            return_uncertainty: Whether to return uncertainty metrics
            
        Returns:
            ClassificationResult or detailed prediction dictionary
            
        Raises:
            ValueError: If no model is loaded or input shape is invalid
        """
        if self.model is None:
            raise ValueError("No model loaded. Build or load a model first.")
        
        # Validate input shape
        expected_shape = self._get_expected_input_shape()
        if not self._validate_input_shape(input_data.shape, expected_shape):
            raise ValueError(
                f"Invalid input shape: {input_data.shape}. "
                f"Expected: {expected_shape}"
            )
        
        # Ensure batch dimension
        if len(input_data.shape) == len(expected_shape):
            input_data = np.expand_dims(input_data, axis=0)
        
        # Measure inference time
        start_time = time.time()
        
        try:
            # Make prediction
            predictions = self.model.predict(input_data, verbose=0)
            inference_time = time.time() - start_time
            
            # Track performance
            self.inference_times.append(inference_time)
            
            # Process predictions
            batch_size = predictions.shape[0]
            results = []
            
            for i in range(batch_size):
                pred_probs = predictions[i]
                predicted_class_idx = np.argmax(pred_probs)
                confidence = float(pred_probs[predicted_class_idx])
                
                # Calculate uncertainty metrics
                uncertainty_metrics = self._calculate_uncertainty(pred_probs)
                
                # Determine affected limbs (simplified heuristic)
                affected_limbs = self._determine_affected_limbs(
                    predicted_class_idx, confidence
                )
                
                # Create classification result
                result = ClassificationResult(
                    abnormality_type=self.class_labels[predicted_class_idx],
                    confidence=confidence,
                    severity_score=self._calculate_severity_score(pred_probs),
                    affected_limbs=affected_limbs
                )
                
                # Store prediction history
                self.prediction_history.append({
                    'timestamp': time.time(),
                    'result': result,
                    'inference_time': inference_time,
                    'uncertainty': uncertainty_metrics
                })
                
                # Return detailed results if requested
                if return_probabilities or return_uncertainty:
                    detailed_result = {
                        'classification_result': result,
                        'inference_time_ms': inference_time * 1000
                    }
                    
                    if return_probabilities:
                        detailed_result['class_probabilities'] = {
                            label: float(prob) 
                            for label, prob in zip(self.class_labels, pred_probs)
                        }
                    
                    if return_uncertainty:
                        detailed_result['uncertainty_metrics'] = uncertainty_metrics
                    
                    results.append(detailed_result)
                else:
                    results.append(result)
            
            # Return single result if batch size is 1
            return results[0] if len(results) == 1 else results
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_proba(self, input_data: np.ndarray) -> Dict[str, float]:
        """
        Get class probabilities for input data.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Dictionary mapping class labels to probabilities
        """
        result = self.predict(
            input_data, 
            return_probabilities=True, 
            return_uncertainty=False
        )
        
        if isinstance(result, dict):
            return result['class_probabilities']
        else:
            # Fallback: make another prediction
            predictions = self.model.predict(
                np.expand_dims(input_data, axis=0) if len(input_data.shape) == len(self._get_expected_input_shape()) else input_data,
                verbose=0
            )
            return {
                label: float(prob) 
                for label, prob in zip(self.class_labels, predictions[0])
            }
    
    def is_prediction_reliable(self, 
                             classification_result: ClassificationResult,
                             uncertainty_metrics: Optional[Dict[str, float]] = None) -> bool:
        """
        Determine if a prediction is reliable based on confidence and uncertainty.
        
        Args:
            classification_result: Classification result to evaluate
            uncertainty_metrics: Optional uncertainty metrics
            
        Returns:
            True if prediction is reliable, False otherwise
        """
        # Check confidence threshold
        if classification_result.confidence < self.confidence_threshold:
            return False
        
        # Check uncertainty threshold if available
        if uncertainty_metrics:
            entropy = uncertainty_metrics.get('entropy', 0)
            if entropy > self.uncertainty_threshold:
                return False
        
        # Additional reliability checks
        if classification_result.abnormality_type == 'normal':
            # Normal gait should have high confidence
            return classification_result.confidence > 0.8
        else:
            # Abnormal gait detection should be more conservative
            return classification_result.confidence > self.confidence_threshold
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the classifier.
        
        Returns:
            Dictionary containing performance statistics
        """
        if not self.inference_times:
            return {
                'avg_inference_time_ms': 0,
                'min_inference_time_ms': 0,
                'max_inference_time_ms': 0,
                'total_predictions': 0,
                'fps_capability': 0
            }
        
        inference_times_ms = [t * 1000 for t in self.inference_times]
        avg_inference_time = np.mean(self.inference_times)
        
        return {
            'avg_inference_time_ms': np.mean(inference_times_ms),
            'min_inference_time_ms': np.min(inference_times_ms),
            'max_inference_time_ms': np.max(inference_times_ms),
            'std_inference_time_ms': np.std(inference_times_ms),
            'total_predictions': len(self.inference_times),
            'fps_capability': 1.0 / avg_inference_time if avg_inference_time > 0 else float('inf'),
            'architecture_type': self.architecture_type,
            'model_parameters': self.model.count_params() if self.model else 0
        }
    
    def reset_performance_tracking(self) -> None:
        """Reset performance tracking metrics."""
        self.inference_times.clear()
        self.prediction_history.clear()
        self.logger.info("Performance tracking reset")
    
    def _filter_model_params(self, config_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter configuration parameters based on architecture type.
        
        Args:
            config_params: Raw configuration parameters
            
        Returns:
            Filtered parameters suitable for the architecture
        """
        filtered = {}
        
        if self.architecture_type == '3dcnn':
            # Map config parameters to 3D-CNN function parameters
            param_mapping = {
                'input_shape': 'input_shape',
                'num_classes': 'num_classes',
                'use_mixed_precision': 'use_mixed_precision'
            }
            
        elif self.architecture_type == 'lstm':
            # Map config parameters to LSTM function parameters
            param_mapping = {
                'input_shape': 'input_shape',
                'num_classes': 'num_classes',
                'units': 'lstm_units',
                'attention': 'attention_dim',
                'use_mixed_precision': 'use_mixed_precision'
            }
            
        elif self.architecture_type == 'hybrid':
            # Map config parameters to Hybrid function parameters
            param_mapping = {
                'frame_shape': 'frame_shape',
                'sequence_length': 'sequence_length',
                'num_classes': 'num_classes',
                'lstm_units': 'lstm_units',
                'use_mixed_precision': 'use_mixed_precision'
            }
        
        # Apply parameter mapping
        for config_key, param_key in param_mapping.items():
            if config_key in config_params:
                filtered[param_key] = config_params[config_key]
        
        return filtered
    
    def _get_expected_input_shape(self) -> Tuple[int, ...]:
        """Get expected input shape for the current architecture."""
        if self.model is None:
            # Return default shapes based on architecture type
            if self.architecture_type == '3dcnn':
                return (16, 224, 224, 3)  # (frames, height, width, channels)
            elif self.architecture_type == 'lstm':
                return (30, 66)  # (sequence_length, features)
            elif self.architecture_type == 'hybrid':
                return (16, 224, 224, 3)  # (frames, height, width, channels)
        
        # Get from model input shape
        input_shape = self.model.input_shape
        return input_shape[1:]  # Remove batch dimension
    
    def _validate_input_shape(self, 
                             input_shape: Tuple[int, ...], 
                             expected_shape: Tuple[int, ...]) -> bool:
        """Validate input shape against expected shape."""
        # Allow batch dimension
        if len(input_shape) == len(expected_shape) + 1:
            input_shape = input_shape[1:]
        
        # Check if shapes match (allowing for variable sequence lengths in some cases)
        if len(input_shape) != len(expected_shape):
            return False
        
        # For LSTM, allow variable sequence length
        if self.architecture_type == 'lstm' and len(input_shape) == 2:
            return input_shape[1] == expected_shape[1]  # Check feature dimension
        
        # For other architectures, check exact match
        return input_shape == expected_shape
    
    def _calculate_uncertainty(self, probabilities: np.ndarray) -> Dict[str, float]:
        """
        Calculate uncertainty metrics from prediction probabilities.
        
        Args:
            probabilities: Class probabilities
            
        Returns:
            Dictionary of uncertainty metrics
        """
        # Entropy (higher = more uncertain)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
        
        # Max probability (higher = more certain)
        max_prob = np.max(probabilities)
        
        # Margin (difference between top two predictions)
        sorted_probs = np.sort(probabilities)[::-1]
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        
        # Gini coefficient (measure of inequality in probabilities)
        gini = 1 - np.sum(probabilities ** 2)
        
        return {
            'entropy': float(entropy),
            'max_probability': float(max_prob),
            'margin': float(margin),
            'gini_coefficient': float(gini),
            'uncertainty_score': float(entropy / np.log(len(probabilities)))  # Normalized entropy
        }
    
    def _calculate_severity_score(self, probabilities: np.ndarray) -> float:
        """
        Calculate severity score based on prediction probabilities.
        
        Args:
            probabilities: Class probabilities
            
        Returns:
            Severity score (0.0 = normal, 1.0 = severe abnormality)
        """
        # Assume first class is 'normal' and others are abnormalities
        normal_prob = probabilities[0]
        abnormal_probs = probabilities[1:]
        
        # Severity is inverse of normal probability, weighted by abnormality confidence
        if len(abnormal_probs) > 0:
            max_abnormal_prob = np.max(abnormal_probs)
            severity = (1 - normal_prob) * max_abnormal_prob
        else:
            severity = 1 - normal_prob
        
        return float(np.clip(severity, 0.0, 1.0))
    
    def _determine_affected_limbs(self, 
                                 predicted_class_idx: int, 
                                 confidence: float) -> List[str]:
        """
        Determine affected limbs based on predicted abnormality type.
        
        This is a simplified heuristic that can be enhanced with more
        sophisticated analysis of pose data.
        
        Args:
            predicted_class_idx: Index of predicted class
            confidence: Prediction confidence
            
        Returns:
            List of affected limbs
        """
        if predicted_class_idx == 0 or confidence < 0.5:  # Normal or low confidence
            return []
        
        abnormality_type = self.class_labels[predicted_class_idx]
        
        # Simple heuristic mapping (can be enhanced with pose analysis)
        limb_mapping = {
            'limping': ['left_leg', 'right_leg'],
            'shuffling': ['left_leg', 'right_leg'],
            'irregular_stride': ['left_leg', 'right_leg'],
            'balance_issues': ['left_leg', 'right_leg', 'trunk']
        }
        
        return limb_mapping.get(abnormality_type, ['unknown'])
    
    def __str__(self) -> str:
        """String representation of the classifier."""
        model_status = "loaded" if self.model is not None else "not loaded"
        return (
            f"GaitClassifier(architecture={self.architecture_type}, "
            f"classes={len(self.class_labels)}, model={model_status})"
        )
    
    def __repr__(self) -> str:
        """Detailed representation of the classifier."""
        return (
            f"GaitClassifier(architecture_type='{self.architecture_type}', "
            f"num_classes={self.num_classes}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"uncertainty_threshold={self.uncertainty_threshold}, "
            f"model_loaded={self.model is not None})"
        )