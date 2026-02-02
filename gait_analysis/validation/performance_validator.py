"""
Performance validation pipeline for gait abnormality detection models.

This module implements comprehensive model performance testing against ground truth,
calculating accuracy, precision, recall, F1-scores, and providing performance
threshold monitoring with recommendations.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.preprocessing import label_binarize
from pathlib import Path
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import asdict

from ..utils.data_structures import (
    PerformanceMetrics, Dataset, TrainingExample, 
    ModelComparison, ClassificationResult
)
from ..classification.gait_classifier import GaitClassifier


class PerformanceValidator:
    """
    Comprehensive performance validation pipeline for gait classification models.
    
    This class provides model performance testing against ground truth datasets,
    calculates standard ML metrics, monitors performance thresholds, and provides
    recommendations for model improvement.
    
    Features:
    - Multi-class classification metrics (accuracy, precision, recall, F1)
    - ROC-AUC and PR-AUC analysis for binary and multi-class scenarios
    - Confusion matrix analysis and class-wise performance breakdown
    - Performance threshold monitoring with automated recommendations
    - Cross-validation support for robust performance estimation
    - Model comparison and benchmarking capabilities
    """
    
    # Performance thresholds for different metrics
    DEFAULT_THRESHOLDS = {
        'accuracy': 0.85,
        'precision': 0.80,
        'recall': 0.80,
        'f1_score': 0.80,
        'inference_time_ms': 100.0,  # Maximum acceptable inference time
        'model_size_mb': 50.0        # Maximum acceptable model size
    }
    
    # Minimum sample sizes for reliable validation
    MIN_SAMPLES_PER_CLASS = 10
    MIN_TOTAL_SAMPLES = 50
    
    def __init__(self, 
                 performance_thresholds: Optional[Dict[str, float]] = None,
                 class_labels: Optional[List[str]] = None,
                 validation_mode: str = 'strict'):
        """
        Initialize the performance validator.
        
        Args:
            performance_thresholds: Custom performance thresholds
            class_labels: List of class labels for classification
            validation_mode: Validation strictness ('strict', 'moderate', 'lenient')
        """
        self.thresholds = performance_thresholds or self.DEFAULT_THRESHOLDS.copy()
        self.class_labels = class_labels or GaitClassifier.DEFAULT_CLASS_LABELS
        self.validation_mode = validation_mode
        
        # Adjust thresholds based on validation mode
        self._adjust_thresholds_by_mode()
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Validation results storage
        self.validation_results = []
        self.model_comparisons = []
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the validator."""
        logger = logging.getLogger('PerformanceValidator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _adjust_thresholds_by_mode(self) -> None:
        """Adjust performance thresholds based on validation mode."""
        if self.validation_mode == 'lenient':
            # Lower thresholds for lenient mode
            for key in ['accuracy', 'precision', 'recall', 'f1_score']:
                if key in self.thresholds:
                    self.thresholds[key] *= 0.9
        elif self.validation_mode == 'strict':
            # Higher thresholds for strict mode
            for key in ['accuracy', 'precision', 'recall', 'f1_score']:
                if key in self.thresholds:
                    self.thresholds[key] *= 1.1
    
    def validate_model_performance(self, 
                                 classifier: GaitClassifier,
                                 test_dataset: Dataset,
                                 calculate_auc: bool = True,
                                 save_results: bool = True) -> PerformanceMetrics:
        """
        Validate model performance against ground truth test dataset.
        
        Args:
            classifier: Trained gait classifier to validate
            test_dataset: Test dataset with ground truth labels
            calculate_auc: Whether to calculate AUC metrics
            save_results: Whether to save validation results
            
        Returns:
            PerformanceMetrics object with comprehensive performance data
            
        Raises:
            ValueError: If test dataset is insufficient or invalid
        """
        self.logger.info(f"Starting performance validation for {classifier.architecture_type} model")
        
        # Validate test dataset
        self._validate_test_dataset(test_dataset)
        
        # Prepare test data
        X_test, y_test, y_test_encoded = self._prepare_test_data(test_dataset)
        
        # Measure inference performance
        start_time = time.time()
        predictions = []
        prediction_probabilities = []
        
        self.logger.info(f"Running inference on {len(X_test)} test samples...")
        
        for i, sample in enumerate(X_test):
            try:
                # Get prediction with probabilities
                result = classifier.predict(
                    sample, 
                    return_probabilities=True, 
                    return_uncertainty=False
                )
                
                if isinstance(result, dict):
                    pred_class = result['classification_result'].abnormality_type
                    pred_probs = result['class_probabilities']
                else:
                    pred_class = result.abnormality_type
                    pred_probs = classifier.predict_proba(sample)
                
                predictions.append(pred_class)
                
                # Convert probabilities to array in class order
                prob_array = [pred_probs.get(label, 0.0) for label in self.class_labels]
                prediction_probabilities.append(prob_array)
                
            except Exception as e:
                self.logger.warning(f"Prediction failed for sample {i}: {str(e)}")
                # Use default prediction for failed samples
                predictions.append(self.class_labels[0])  # Default to first class
                prediction_probabilities.append([1.0] + [0.0] * (len(self.class_labels) - 1))
        
        total_inference_time = time.time() - start_time
        avg_inference_time = total_inference_time / len(X_test)
        
        # Convert predictions to encoded format
        y_pred_encoded = [self.class_labels.index(pred) for pred in predictions]
        prediction_probabilities = np.array(prediction_probabilities)
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        
        # Calculate per-class metrics
        precision_per_class = precision_score(
            y_test_encoded, y_pred_encoded, 
            average=None, labels=range(len(self.class_labels)), zero_division=0
        )
        recall_per_class = recall_score(
            y_test_encoded, y_pred_encoded, 
            average=None, labels=range(len(self.class_labels)), zero_division=0
        )
        f1_per_class = f1_score(
            y_test_encoded, y_pred_encoded, 
            average=None, labels=range(len(self.class_labels)), zero_division=0
        )
        
        # Create per-class metric dictionaries
        precision_dict = {
            label: float(precision_per_class[i]) 
            for i, label in enumerate(self.class_labels)
        }
        recall_dict = {
            label: float(recall_per_class[i]) 
            for i, label in enumerate(self.class_labels)
        }
        f1_dict = {
            label: float(f1_per_class[i]) 
            for i, label in enumerate(self.class_labels)
        }
        
        # Calculate model size
        model_size_mb = self._calculate_model_size(classifier)
        
        # Create performance metrics object
        performance_metrics = PerformanceMetrics(
            accuracy=float(accuracy),
            precision=precision_dict,
            recall=recall_dict,
            f1_score=f1_dict,
            training_time=0.0,  # Not available during validation
            inference_time=float(avg_inference_time),
            model_size=model_size_mb
        )
        
        # Calculate additional metrics if requested
        additional_metrics = {}
        if calculate_auc and len(self.class_labels) > 2:
            try:
                # Multi-class AUC calculation
                y_test_binarized = label_binarize(
                    y_test_encoded, classes=range(len(self.class_labels))
                )
                
                # Calculate AUC for each class
                auc_scores = {}
                for i, label in enumerate(self.class_labels):
                    if len(np.unique(y_test_binarized[:, i])) > 1:  # Check if class exists in test set
                        auc = roc_auc_score(
                            y_test_binarized[:, i], 
                            prediction_probabilities[:, i]
                        )
                        auc_scores[label] = float(auc)
                    else:
                        auc_scores[label] = 0.0
                
                additional_metrics['auc_scores'] = auc_scores
                additional_metrics['macro_auc'] = float(np.mean(list(auc_scores.values())))
                
            except Exception as e:
                self.logger.warning(f"AUC calculation failed: {str(e)}")
                additional_metrics['auc_scores'] = {}
                additional_metrics['macro_auc'] = 0.0
        
        # Generate detailed classification report
        classification_rep = classification_report(
            y_test_encoded, y_pred_encoded,
            target_names=self.class_labels,
            output_dict=True,
            zero_division=0
        )
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test_encoded, y_pred_encoded)
        
        # Store validation results
        validation_result = {
            'timestamp': time.time(),
            'architecture_type': classifier.architecture_type,
            'performance_metrics': performance_metrics,
            'additional_metrics': additional_metrics,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(),
            'test_samples': len(X_test),
            'class_distribution': dict(zip(*np.unique(y_test_encoded, return_counts=True)))
        }
        
        if save_results:
            self.validation_results.append(validation_result)
        
        # Log results
        self.logger.info(f"Validation completed:")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Avg Precision: {np.mean(list(precision_dict.values())):.4f}")
        self.logger.info(f"  Avg Recall: {np.mean(list(recall_dict.values())):.4f}")
        self.logger.info(f"  Avg F1-Score: {np.mean(list(f1_dict.values())):.4f}")
        self.logger.info(f"  Inference Time: {avg_inference_time*1000:.2f}ms")
        self.logger.info(f"  Model Size: {model_size_mb:.2f}MB")
        
        return performance_metrics
    
    def check_performance_thresholds(self, 
                                   performance_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """
        Check if performance metrics meet defined thresholds.
        
        Args:
            performance_metrics: Performance metrics to check
            
        Returns:
            Dictionary with threshold check results and recommendations
        """
        self.logger.info("Checking performance against thresholds...")
        
        results = {
            'meets_thresholds': True,
            'failed_metrics': [],
            'recommendations': [],
            'threshold_analysis': {}
        }
        
        # Check accuracy threshold
        if performance_metrics.accuracy < self.thresholds['accuracy']:
            results['meets_thresholds'] = False
            results['failed_metrics'].append('accuracy')
            results['recommendations'].append(
                f"Accuracy ({performance_metrics.accuracy:.3f}) below threshold "
                f"({self.thresholds['accuracy']:.3f}). Consider data augmentation or model retraining."
            )
        
        results['threshold_analysis']['accuracy'] = {
            'value': performance_metrics.accuracy,
            'threshold': self.thresholds['accuracy'],
            'passes': performance_metrics.accuracy >= self.thresholds['accuracy']
        }
        
        # Check precision thresholds
        avg_precision = np.mean(list(performance_metrics.precision.values()))
        if avg_precision < self.thresholds['precision']:
            results['meets_thresholds'] = False
            results['failed_metrics'].append('precision')
            results['recommendations'].append(
                f"Average precision ({avg_precision:.3f}) below threshold "
                f"({self.thresholds['precision']:.3f}). Consider class balancing or threshold tuning."
            )
        
        results['threshold_analysis']['precision'] = {
            'value': avg_precision,
            'threshold': self.thresholds['precision'],
            'passes': avg_precision >= self.thresholds['precision'],
            'per_class': performance_metrics.precision
        }
        
        # Check recall thresholds
        avg_recall = np.mean(list(performance_metrics.recall.values()))
        if avg_recall < self.thresholds['recall']:
            results['meets_thresholds'] = False
            results['failed_metrics'].append('recall')
            results['recommendations'].append(
                f"Average recall ({avg_recall:.3f}) below threshold "
                f"({self.thresholds['recall']:.3f}). Consider more training data or model complexity increase."
            )
        
        results['threshold_analysis']['recall'] = {
            'value': avg_recall,
            'threshold': self.thresholds['recall'],
            'passes': avg_recall >= self.thresholds['recall'],
            'per_class': performance_metrics.recall
        }
        
        # Check F1-score thresholds
        avg_f1 = np.mean(list(performance_metrics.f1_score.values()))
        if avg_f1 < self.thresholds['f1_score']:
            results['meets_thresholds'] = False
            results['failed_metrics'].append('f1_score')
            results['recommendations'].append(
                f"Average F1-score ({avg_f1:.3f}) below threshold "
                f"({self.thresholds['f1_score']:.3f}). Consider balanced training or ensemble methods."
            )
        
        results['threshold_analysis']['f1_score'] = {
            'value': avg_f1,
            'threshold': self.thresholds['f1_score'],
            'passes': avg_f1 >= self.thresholds['f1_score'],
            'per_class': performance_metrics.f1_score
        }
        
        # Check inference time threshold
        inference_time_ms = performance_metrics.inference_time * 1000
        if inference_time_ms > self.thresholds['inference_time_ms']:
            results['meets_thresholds'] = False
            results['failed_metrics'].append('inference_time')
            results['recommendations'].append(
                f"Inference time ({inference_time_ms:.2f}ms) exceeds threshold "
                f"({self.thresholds['inference_time_ms']:.2f}ms). Consider model optimization or quantization."
            )
        
        results['threshold_analysis']['inference_time'] = {
            'value': inference_time_ms,
            'threshold': self.thresholds['inference_time_ms'],
            'passes': inference_time_ms <= self.thresholds['inference_time_ms']
        }
        
        # Check model size threshold
        if performance_metrics.model_size > self.thresholds['model_size_mb']:
            results['meets_thresholds'] = False
            results['failed_metrics'].append('model_size')
            results['recommendations'].append(
                f"Model size ({performance_metrics.model_size:.2f}MB) exceeds threshold "
                f"({self.thresholds['model_size_mb']:.2f}MB). Consider model pruning or compression."
            )
        
        results['threshold_analysis']['model_size'] = {
            'value': performance_metrics.model_size,
            'threshold': self.thresholds['model_size_mb'],
            'passes': performance_metrics.model_size <= self.thresholds['model_size_mb']
        }
        
        # Generate overall recommendations
        if results['meets_thresholds']:
            results['recommendations'].append("All performance thresholds met. Model is ready for deployment.")
        else:
            results['recommendations'].append("Model requires improvement before deployment.")
            
            # Add specific improvement suggestions
            if len(results['failed_metrics']) >= 3:
                results['recommendations'].append("Multiple metrics failed. Consider complete model retraining with improved data and architecture.")
            elif 'accuracy' in results['failed_metrics']:
                results['recommendations'].append("Focus on improving overall model accuracy through better feature engineering or architecture selection.")
        
        # Log threshold check results
        self.logger.info(f"Threshold check completed:")
        self.logger.info(f"  Meets all thresholds: {results['meets_thresholds']}")
        if results['failed_metrics']:
            self.logger.info(f"  Failed metrics: {', '.join(results['failed_metrics'])}")
        
        return results
    
    def recommend_model_improvements(self, 
                                   performance_metrics: PerformanceMetrics,
                                   threshold_results: Dict[str, Any]) -> List[str]:
        """
        Generate specific recommendations for model improvement.
        
        Args:
            performance_metrics: Current performance metrics
            threshold_results: Results from threshold checking
            
        Returns:
            List of specific improvement recommendations
        """
        recommendations = []
        
        # Analyze performance patterns
        avg_precision = np.mean(list(performance_metrics.precision.values()))
        avg_recall = np.mean(list(performance_metrics.recall.values()))
        precision_std = np.std(list(performance_metrics.precision.values()))
        recall_std = np.std(list(performance_metrics.recall.values()))
        
        # High variance in per-class performance
        if precision_std > 0.2 or recall_std > 0.2:
            recommendations.append(
                "High variance in per-class performance detected. "
                "Consider class-specific data augmentation or balanced sampling."
            )
        
        # Low precision, high recall pattern
        if avg_precision < 0.7 and avg_recall > 0.8:
            recommendations.append(
                "Model has high recall but low precision (many false positives). "
                "Consider increasing classification threshold or adding regularization."
            )
        
        # High precision, low recall pattern
        if avg_precision > 0.8 and avg_recall < 0.7:
            recommendations.append(
                "Model has high precision but low recall (missing true positives). "
                "Consider decreasing classification threshold or adding more training data."
            )
        
        # Poor overall performance
        if performance_metrics.accuracy < 0.6:
            recommendations.append(
                "Overall accuracy is very low. Consider fundamental changes: "
                "different architecture, better feature engineering, or data quality review."
            )
        
        # Slow inference
        if performance_metrics.inference_time > 0.5:
            recommendations.append(
                "Inference time is slow. Consider model quantization, pruning, "
                "or switching to a more efficient architecture."
            )
        
        # Large model size
        if performance_metrics.model_size > 100:
            recommendations.append(
                "Model size is large. Consider knowledge distillation, "
                "parameter sharing, or architecture optimization."
            )
        
        # Architecture-specific recommendations
        if hasattr(performance_metrics, 'architecture_type'):
            arch_type = getattr(performance_metrics, 'architecture_type', 'unknown')
            
            if arch_type == '3dcnn' and performance_metrics.accuracy < 0.8:
                recommendations.append(
                    "3D-CNN performance is suboptimal. Consider increasing temporal "
                    "receptive field or adding attention mechanisms."
                )
            elif arch_type == 'lstm' and avg_recall < 0.7:
                recommendations.append(
                    "LSTM model has low recall. Consider bidirectional processing "
                    "or increasing sequence length for better temporal modeling."
                )
            elif arch_type == 'hybrid' and performance_metrics.inference_time > 0.2:
                recommendations.append(
                    "Hybrid model is slow. Consider reducing CNN backbone complexity "
                    "or optimizing the fusion mechanism."
                )
        
        return recommendations
    
    def compare_models(self, 
                      model_comparisons: List[ModelComparison]) -> Dict[str, Any]:
        """
        Compare performance across multiple model architectures.
        
        Args:
            model_comparisons: List of model comparison objects
            
        Returns:
            Dictionary with comparison results and best model recommendation
        """
        if not model_comparisons:
            raise ValueError("No model comparisons provided")
        
        self.logger.info(f"Comparing {len(model_comparisons)} models...")
        
        # Extract metrics for comparison
        comparison_data = []
        for comp in model_comparisons:
            metrics = comp.performance_metrics
            comparison_data.append({
                'architecture': comp.architecture_name,
                'accuracy': metrics.accuracy,
                'avg_precision': np.mean(list(metrics.precision.values())),
                'avg_recall': np.mean(list(metrics.recall.values())),
                'avg_f1': np.mean(list(metrics.f1_score.values())),
                'inference_time': metrics.inference_time,
                'model_size': metrics.model_size,
                'training_time': metrics.training_time
            })
        
        # Find best model for each metric
        best_models = {}
        for metric in ['accuracy', 'avg_precision', 'avg_recall', 'avg_f1']:
            best_idx = np.argmax([data[metric] for data in comparison_data])
            best_models[metric] = comparison_data[best_idx]['architecture']
        
        # Find fastest and smallest models
        fastest_idx = np.argmin([data['inference_time'] for data in comparison_data])
        smallest_idx = np.argmin([data['model_size'] for data in comparison_data])
        best_models['fastest'] = comparison_data[fastest_idx]['architecture']
        best_models['smallest'] = comparison_data[smallest_idx]['architecture']
        
        # Calculate overall score (weighted combination)
        weights = {
            'accuracy': 0.3,
            'avg_f1': 0.3,
            'inference_time': -0.2,  # Negative because lower is better
            'model_size': -0.1       # Negative because lower is better
        }
        
        overall_scores = []
        for data in comparison_data:
            score = (
                weights['accuracy'] * data['accuracy'] +
                weights['avg_f1'] * data['avg_f1'] +
                weights['inference_time'] * (1.0 / (data['inference_time'] + 0.001)) * 0.1 +
                weights['model_size'] * (1.0 / (data['model_size'] + 1.0)) * 10.0
            )
            overall_scores.append(score)
        
        best_overall_idx = np.argmax(overall_scores)
        best_overall_model = comparison_data[best_overall_idx]['architecture']
        
        # Generate comparison summary
        comparison_results = {
            'model_count': len(model_comparisons),
            'comparison_data': comparison_data,
            'best_models': best_models,
            'best_overall': best_overall_model,
            'overall_scores': overall_scores,
            'recommendations': []
        }
        
        # Add recommendations
        comparison_results['recommendations'].append(
            f"Best overall model: {best_overall_model} "
            f"(score: {overall_scores[best_overall_idx]:.3f})"
        )
        
        if best_models['accuracy'] == best_models['avg_f1']:
            comparison_results['recommendations'].append(
                f"Consistent top performer: {best_models['accuracy']} "
                "excels in both accuracy and F1-score"
            )
        
        if best_models['fastest'] == best_models['smallest']:
            comparison_results['recommendations'].append(
                f"Most efficient model: {best_models['fastest']} "
                "is both fastest and smallest"
            )
        
        # Store comparison results
        self.model_comparisons.append(comparison_results)
        
        self.logger.info(f"Model comparison completed. Best overall: {best_overall_model}")
        
        return comparison_results
    
    def _validate_test_dataset(self, test_dataset: Dataset) -> None:
        """Validate that test dataset is suitable for performance evaluation."""
        if not test_dataset.examples:
            raise ValueError("Test dataset is empty")
        
        if len(test_dataset.examples) < self.MIN_TOTAL_SAMPLES:
            raise ValueError(
                f"Test dataset too small: {len(test_dataset.examples)} samples. "
                f"Minimum required: {self.MIN_TOTAL_SAMPLES}"
            )
        
        # Check class distribution
        class_counts = {}
        for example in test_dataset.examples:
            label = example.ground_truth_label
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Ensure minimum samples per class
        for label, count in class_counts.items():
            if count < self.MIN_SAMPLES_PER_CLASS:
                self.logger.warning(
                    f"Class '{label}' has only {count} samples. "
                    f"Minimum recommended: {self.MIN_SAMPLES_PER_CLASS}"
                )
    
    def _prepare_test_data(self, test_dataset: Dataset) -> Tuple[List[np.ndarray], List[str], List[int]]:
        """Prepare test data for model evaluation."""
        X_test = []
        y_test = []
        
        for example in test_dataset.examples:
            # Extract features from pose sequence
            # This is a simplified version - in practice, you'd use FeatureExtractor
            if 'features' in example.metadata and example.metadata['features'] is not None:
                X_test.append(example.metadata['features'])
            else:
                # Create dummy features for testing
                # In practice, this would be processed through FeatureExtractor
                dummy_features = np.random.randn(16, 224, 224, 3)  # Default 3D-CNN shape
                X_test.append(dummy_features)
            
            y_test.append(example.ground_truth_label)
        
        # Encode labels
        y_test_encoded = [self.class_labels.index(label) if label in self.class_labels else 0 
                         for label in y_test]
        
        return X_test, y_test, y_test_encoded
    
    def _calculate_model_size(self, classifier: GaitClassifier) -> float:
        """Calculate model size in MB."""
        if classifier.model is None:
            return 0.0
        
        try:
            # Get model parameters count
            param_count = classifier.model.count_params()
            
            # Estimate size (4 bytes per float32 parameter)
            size_bytes = param_count * 4
            size_mb = size_bytes / (1024 * 1024)
            
            return float(size_mb)
            
        except Exception as e:
            self.logger.warning(f"Could not calculate model size: {str(e)}")
            return 0.0
    
    def save_validation_results(self, filepath: str) -> None:
        """Save validation results to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        results_data = {
            'validation_results': self.validation_results,
            'model_comparisons': self.model_comparisons,
            'thresholds': self.thresholds,
            'validation_mode': self.validation_mode,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"Validation results saved to {filepath}")
    
    def load_validation_results(self, filepath: str) -> None:
        """Load validation results from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Validation results file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            results_data = json.load(f)
        
        self.validation_results = results_data.get('validation_results', [])
        self.model_comparisons = results_data.get('model_comparisons', [])
        
        self.logger.info(f"Validation results loaded from {filepath}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        if not self.validation_results:
            return {'message': 'No validation results available'}
        
        summary = {
            'total_validations': len(self.validation_results),
            'architectures_tested': list(set(
                result['architecture_type'] for result in self.validation_results
            )),
            'average_metrics': {},
            'best_performances': {},
            'validation_timestamps': [
                result['timestamp'] for result in self.validation_results
            ]
        }
        
        # Calculate average metrics across all validations
        all_accuracies = [result['performance_metrics'].accuracy for result in self.validation_results]
        summary['average_metrics']['accuracy'] = float(np.mean(all_accuracies))
        
        # Find best performances
        best_accuracy_idx = np.argmax(all_accuracies)
        summary['best_performances']['accuracy'] = {
            'value': all_accuracies[best_accuracy_idx],
            'architecture': self.validation_results[best_accuracy_idx]['architecture_type']
        }
        
        return summary