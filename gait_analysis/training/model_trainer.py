"""
Model training system with dataset validation and performance tracking.

This module implements the ModelTrainer class that handles training of all three
model architectures with comprehensive dataset validation, error handling, and
performance tracking as specified in Requirements 2.1, 2.3, and 2.4.
"""

import os
import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..models import (
    create_lightweight_3dcnn,
    create_efficient_lstm, 
    create_hybrid_cnn_lstm,
    LightweightCNN3D,
    EfficientLSTM,
    HybridCNNLSTM
)
from ..utils.data_structures import (
    Dataset,
    TrainingExample,
    PerformanceMetrics,
    TrainingHistory,
    ModelComparison
)
from ..utils.config import get_config
from .error_handling import TrainingErrorHandler, RobustTrainingCallback


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    architecture: str  # '3dcnn', 'lstm', 'hybrid'
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    use_mixed_precision: bool = True
    save_best_only: bool = True
    model_save_path: str = 'models/'
    log_dir: str = 'logs/'
    
    # Architecture-specific parameters
    # 3D-CNN parameters
    input_shape_3dcnn: Tuple[int, int, int, int] = (16, 224, 224, 3)
    
    # LSTM parameters  
    input_shape_lstm: Tuple[int, int] = (30, 66)
    lstm_units: int = 64
    attention_dim: int = 64
    
    # Hybrid parameters
    frame_shape: Tuple[int, int, int] = (224, 224, 3)
    sequence_length: int = 16
    spatial_feature_dim: int = 128
    fusion_dim: int = 128
    trainable_backbone: bool = False


class DatasetValidator:
    """Validates dataset format and consistency for training."""
    
    def __init__(self):
        """Initialize dataset validator."""
        self.logger = logging.getLogger(__name__)
        
    def validate_dataset(self, dataset: Dataset) -> Tuple[bool, List[str]]:
        """
        Validate dataset format and labeling consistency.
        
        Validates Requirements 2.1: Dataset format validation and consistency checking
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Check if dataset has examples
            if not dataset.examples:
                errors.append("Dataset is empty - no training examples found")
                return False, errors
            
            # Validate class distribution
            if not dataset.class_distribution:
                errors.append("Dataset class distribution is empty")
            else:
                total_examples = sum(dataset.class_distribution.values())
                if total_examples != len(dataset.examples):
                    errors.append(
                        f"Class distribution count ({total_examples}) doesn't match "
                        f"number of examples ({len(dataset.examples)})"
                    )
            
            # Check validation split
            if not 0.0 < dataset.validation_split < 1.0:
                errors.append(
                    f"Invalid validation split: {dataset.validation_split}. "
                    "Must be between 0.0 and 1.0"
                )
            
            # Validate individual examples
            label_consistency_errors = self._validate_examples(dataset.examples)
            errors.extend(label_consistency_errors)
            
            # Check minimum examples per class
            min_examples_per_class = 5
            for class_name, count in dataset.class_distribution.items():
                if count < min_examples_per_class:
                    errors.append(
                        f"Class '{class_name}' has only {count} examples. "
                        f"Minimum required: {min_examples_per_class}"
                    )
            
            # Check class balance (warn if severely imbalanced)
            self._check_class_balance(dataset.class_distribution)
            
            is_valid = len(errors) == 0
            return is_valid, errors
            
        except Exception as e:
            errors.append(f"Unexpected error during dataset validation: {str(e)}")
            return False, errors
    
    def _validate_examples(self, examples: List[TrainingExample]) -> List[str]:
        """Validate individual training examples."""
        errors = []
        
        for i, example in enumerate(examples):
            try:
                # Check video path exists
                if not example.video_path:
                    errors.append(f"Example {i}: Empty video path")
                elif not os.path.exists(example.video_path):
                    errors.append(f"Example {i}: Video file not found: {example.video_path}")
                
                # Check ground truth label
                if not example.ground_truth_label:
                    errors.append(f"Example {i}: Empty ground truth label")
                
                # Check severity score range
                if not 0.0 <= example.severity_score <= 1.0:
                    errors.append(
                        f"Example {i}: Invalid severity score {example.severity_score}. "
                        "Must be between 0.0 and 1.0"
                    )
                
                # Validate pose sequence if present
                if example.pose_sequence:
                    pose_errors = self._validate_pose_sequence(example.pose_sequence, i)
                    errors.extend(pose_errors)
                    
            except Exception as e:
                errors.append(f"Example {i}: Validation error - {str(e)}")
        
        return errors
    
    def _validate_pose_sequence(self, pose_sequence, example_idx: int) -> List[str]:
        """Validate pose sequence data."""
        errors = []
        
        try:
            # Check if keypoints exist
            if not pose_sequence.keypoints:
                errors.append(f"Example {example_idx}: Empty pose keypoints")
                return errors
            
            # Check consistency between frames
            num_frames = len(pose_sequence.keypoints)
            if num_frames == 0:
                errors.append(f"Example {example_idx}: No pose frames found")
                return errors
            
            # Check keypoint count consistency
            expected_keypoints = len(pose_sequence.keypoints[0]) if pose_sequence.keypoints else 0
            for frame_idx, frame_keypoints in enumerate(pose_sequence.keypoints):
                if len(frame_keypoints) != expected_keypoints:
                    errors.append(
                        f"Example {example_idx}, Frame {frame_idx}: "
                        f"Inconsistent keypoint count. Expected {expected_keypoints}, "
                        f"got {len(frame_keypoints)}"
                    )
            
            # Check timestamps consistency
            if len(pose_sequence.timestamps) != num_frames:
                errors.append(
                    f"Example {example_idx}: Timestamp count ({len(pose_sequence.timestamps)}) "
                    f"doesn't match frame count ({num_frames})"
                )
            
            # Check confidence scores
            if len(pose_sequence.confidence_scores) != num_frames:
                errors.append(
                    f"Example {example_idx}: Confidence score count "
                    f"({len(pose_sequence.confidence_scores)}) doesn't match frame count ({num_frames})"
                )
            
        except Exception as e:
            errors.append(f"Example {example_idx}: Pose sequence validation error - {str(e)}")
        
        return errors
    
    def _check_class_balance(self, class_distribution: Dict[str, int]) -> None:
        """Check and warn about class imbalance."""
        if not class_distribution:
            return
        
        counts = list(class_distribution.values())
        max_count = max(counts)
        min_count = min(counts)
        
        # Warn if imbalance ratio > 5:1
        if max_count / min_count > 5:
            self.logger.warning(
                f"Severe class imbalance detected. "
                f"Max class: {max_count}, Min class: {min_count}. "
                f"Consider data augmentation or class weighting."
            )


class ModelTrainer:
    """
    Model trainer with dataset validation and performance tracking.
    
    Handles training of all three model architectures with comprehensive
    error handling, logging, and performance metrics tracking.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize model trainer.
        
        Args:
            config: Training configuration. If None, uses default config.
        """
        self.config = config or TrainingConfig()
        self.validator = DatasetValidator()
        self.logger = self._setup_logging()
        
        # Initialize error handler for comprehensive error handling and logging
        # Implements Requirements 2.4: Error logging and partial progress preservation
        self.error_handler = TrainingErrorHandler(
            log_dir=self.config.log_dir,
            checkpoint_dir=os.path.join(self.config.model_save_path, "checkpoints"),
            max_checkpoints=5
        )
        
        # Training state
        self.model = None
        self.training_history = None
        self.performance_metrics = None
        
        # Create directories
        os.makedirs(self.config.model_save_path, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for training process."""
        logger = logging.getLogger(f"{__name__}.{self.config.architecture}")
        logger.setLevel(logging.INFO)
        
        # Ensure log directory exists
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Create file handler
        log_file = os.path.join(
            self.config.log_dir, 
            f"training_{self.config.architecture}_{int(time.time())}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def validate_and_prepare_dataset(self, dataset: Dataset) -> Tuple[bool, Optional[tf.data.Dataset], Optional[tf.data.Dataset]]:
        """
        Validate dataset and prepare training/validation splits.
        
        Implements Requirements 2.1: Dataset format validation and consistency checking
        
        Args:
            dataset: Dataset to validate and prepare
            
        Returns:
            Tuple of (is_valid, train_dataset, val_dataset)
        """
        self.logger.info("Starting dataset validation...")
        
        # Validate dataset
        is_valid, errors = self.validator.validate_dataset(dataset)
        
        if not is_valid:
            self.logger.error("Dataset validation failed:")
            for error in errors:
                self.logger.error(f"  - {error}")
            return False, None, None
        
        self.logger.info("Dataset validation passed successfully")
        
        try:
            # Prepare data based on architecture
            if self.config.architecture == '3dcnn':
                train_ds, val_ds = self._prepare_3dcnn_data(dataset)
            elif self.config.architecture == 'lstm':
                train_ds, val_ds = self._prepare_lstm_data(dataset)
            elif self.config.architecture == 'hybrid':
                train_ds, val_ds = self._prepare_hybrid_data(dataset)
            else:
                raise ValueError(f"Unsupported architecture: {self.config.architecture}")
            
            self.logger.info("Dataset preparation completed successfully")
            return True, train_ds, val_ds
            
        except Exception as e:
            self.logger.error(f"Dataset preparation failed: {str(e)}")
            return False, None, None
    
    def _prepare_3dcnn_data(self, dataset: Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Prepare data for 3D-CNN training."""
        # This is a placeholder - in real implementation, this would
        # convert video files to 3D tensors
        self.logger.info("Preparing 3D-CNN data format...")
        
        # For now, create dummy data matching the expected shape
        num_examples = len(dataset.examples)
        num_classes = len(dataset.class_distribution)
        
        # Create dummy video data
        X = np.random.random((num_examples,) + self.config.input_shape_3dcnn).astype(np.float32)
        y = np.random.randint(0, num_classes, num_examples)
        y = tf.keras.utils.to_categorical(y, num_classes)
        
        # Split data
        split_idx = int(len(X) * (1 - self.config.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds
    
    def _prepare_lstm_data(self, dataset: Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Prepare data for LSTM training."""
        self.logger.info("Preparing LSTM data format...")
        
        # For now, create dummy pose sequence data
        num_examples = len(dataset.examples)
        num_classes = len(dataset.class_distribution)
        
        # Create dummy pose sequence data
        X = np.random.random((num_examples,) + self.config.input_shape_lstm).astype(np.float32)
        y = np.random.randint(0, num_classes, num_examples)
        y = tf.keras.utils.to_categorical(y, num_classes)
        
        # Split data
        split_idx = int(len(X) * (1 - self.config.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds
    
    def _prepare_hybrid_data(self, dataset: Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Prepare data for hybrid CNN-LSTM training."""
        self.logger.info("Preparing hybrid CNN-LSTM data format...")
        
        # For now, create dummy video sequence data
        num_examples = len(dataset.examples)
        num_classes = len(dataset.class_distribution)
        
        input_shape = (self.config.sequence_length,) + self.config.frame_shape
        
        # Create dummy video sequence data
        X = np.random.random((num_examples,) + input_shape).astype(np.float32)
        y = np.random.randint(0, num_classes, num_examples)
        y = tf.keras.utils.to_categorical(y, num_classes)
        
        # Split data
        split_idx = int(len(X) * (1 - self.config.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds
    
    def create_model(self, num_classes: int) -> Union[LightweightCNN3D, EfficientLSTM, HybridCNNLSTM]:
        """
        Create model based on architecture configuration.
        
        Args:
            num_classes: Number of output classes
            
        Returns:
            Created model instance
        """
        self.logger.info(f"Creating {self.config.architecture} model...")
        
        if self.config.architecture == '3dcnn':
            model = create_lightweight_3dcnn(
                input_shape=self.config.input_shape_3dcnn,
                num_classes=num_classes,
                use_mixed_precision=self.config.use_mixed_precision
            )
        elif self.config.architecture == 'lstm':
            model = create_efficient_lstm(
                input_shape=self.config.input_shape_lstm,
                num_classes=num_classes,
                lstm_units=self.config.lstm_units,
                attention_dim=self.config.attention_dim,
                use_mixed_precision=self.config.use_mixed_precision
            )
        elif self.config.architecture == 'hybrid':
            model = create_hybrid_cnn_lstm(
                frame_shape=self.config.frame_shape,
                sequence_length=self.config.sequence_length,
                num_classes=num_classes,
                spatial_feature_dim=self.config.spatial_feature_dim,
                lstm_units=self.config.lstm_units,
                fusion_dim=self.config.fusion_dim,
                trainable_backbone=self.config.trainable_backbone,
                use_mixed_precision=self.config.use_mixed_precision
            )
        else:
            raise ValueError(f"Unsupported architecture: {self.config.architecture}")
        
        # Recompile with custom learning rate and optimizer
        model.compile_model(
            learning_rate=self.config.learning_rate,
            optimizer=self.config.optimizer
        )
        
        self.model = model
        self.logger.info("Model created and compiled successfully")
        return model
    
    def train_model(self, 
                   dataset: Dataset,
                   save_model: bool = True,
                   resume_from_checkpoint: bool = True) -> Tuple[bool, Optional[TrainingHistory], Optional[PerformanceMetrics]]:
        """
        Train model with comprehensive error handling and performance tracking.
        
        Implements Requirements 2.3: Model saving with performance metrics
        Implements Requirements 2.4: Error logging and partial progress preservation
        
        Args:
            dataset: Training dataset
            save_model: Whether to save the trained model
            resume_from_checkpoint: Whether to attempt resuming from checkpoint
            
        Returns:
            Tuple of (success, training_history, performance_metrics)
        """
        training_context = {
            'dataset': dataset,
            'config': self.config,
            'architecture': self.config.architecture
        }
        
        try:
            self.logger.info("Starting model training with error handling...")
            
            # Check for existing checkpoint if resume is enabled
            checkpoint = None
            if resume_from_checkpoint:
                checkpoint = self.error_handler.load_latest_checkpoint()
                if checkpoint:
                    self.logger.info(f"Found checkpoint at epoch {checkpoint.epoch}")
            
            # Validate and prepare dataset
            is_valid, train_ds, val_ds = self.validate_and_prepare_dataset(dataset)
            if not is_valid:
                error_msg = "Dataset validation failed"
                error = ValueError(error_msg)
                self.error_handler.log_training_error(error, training_context)
                return False, None, None
            
            # Create model
            num_classes = len(dataset.class_distribution)
            model = self.create_model(num_classes)
            training_context['model'] = model
            
            # Resume from checkpoint if available
            start_epoch = 0
            initial_history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
            
            if checkpoint:
                success = self.error_handler.resume_training_from_checkpoint(
                    model, model.model.optimizer, checkpoint
                )
                if success:
                    start_epoch = checkpoint.epoch + 1
                    initial_history = checkpoint.training_history
                    self.logger.info(f"Resumed training from epoch {start_epoch}")
                else:
                    self.logger.warning("Failed to resume from checkpoint, starting fresh")
            
            # Setup callbacks with error handling
            callbacks = self._get_training_callbacks_with_error_handling()
            
            # Record training start time
            training_start_time = time.time()
            
            # Train model with error handling
            self.logger.info(f"Training {self.config.architecture} model for {self.config.epochs} epochs...")
            
            try:
                if hasattr(model, 'train_model'):
                    # Use model-specific training method
                    history = model.train_model(
                        train_dataset=train_ds,
                        val_dataset=val_ds,
                        epochs=self.config.epochs,
                        callbacks=callbacks
                    )
                else:
                    # Use standard Keras fit method
                    history = model.model.fit(
                        train_ds,
                        validation_data=val_ds,
                        epochs=self.config.epochs,
                        initial_epoch=start_epoch,
                        callbacks=callbacks,
                        verbose=1
                    )
                
            except Exception as training_error:
                # Log detailed training error
                self.error_handler.log_training_error(
                    error=training_error,
                    context=training_context,
                    epoch=getattr(training_error, 'epoch', None),
                    batch=getattr(training_error, 'batch', None)
                )
                
                # Attempt to save emergency checkpoint
                try:
                    self.logger.info("Attempting to save emergency checkpoint...")
                    emergency_history = getattr(model.model, 'history', {}).get('history', initial_history)
                    self.error_handler.save_training_checkpoint(
                        model=model,
                        optimizer=model.model.optimizer,
                        epoch=getattr(training_error, 'epoch', start_epoch),
                        batch=getattr(training_error, 'batch', 0),
                        training_history=emergency_history,
                        config=asdict(self.config)
                    )
                except Exception as checkpoint_error:
                    self.logger.error(f"Failed to save emergency checkpoint: {str(checkpoint_error)}")
                
                return False, None, None
            
            # Calculate training time
            training_time = time.time() - training_start_time
            
            # Merge history with initial history if resuming
            final_history = history.history
            if checkpoint and initial_history:
                for key in final_history:
                    if key in initial_history:
                        final_history[key] = initial_history[key] + final_history[key]
            
            # Create training history object
            training_history = TrainingHistory(
                train_loss=final_history.get('loss', []),
                val_loss=final_history.get('val_loss', []),
                train_accuracy=final_history.get('accuracy', []),
                val_accuracy=final_history.get('val_accuracy', []),
                epochs=len(final_history.get('loss', [])),
                best_epoch=np.argmax(final_history.get('val_accuracy', [0])) if final_history.get('val_accuracy') else 0
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                model, val_ds, training_time, training_history
            )
            
            # Save model and metrics if requested
            if save_model:
                self._save_model_with_metrics(model, performance_metrics, training_history)
            
            # Save final checkpoint
            try:
                self.error_handler.save_training_checkpoint(
                    model=model,
                    optimizer=model.model.optimizer,
                    epoch=training_history.epochs - 1,
                    batch=0,
                    training_history=final_history,
                    config=asdict(self.config)
                )
            except Exception as e:
                self.logger.warning(f"Could not save final checkpoint: {str(e)}")
            
            self.training_history = training_history
            self.performance_metrics = performance_metrics
            
            self.logger.info("Model training completed successfully")
            self.logger.info(f"Final validation accuracy: {performance_metrics.accuracy:.4f}")
            
            return True, training_history, performance_metrics
            
        except Exception as e:
            # Log unexpected error
            self.error_handler.log_training_error(e, training_context)
            self.logger.error(f"Unexpected training failure: {str(e)}")
            return False, None, None
    
    def _get_training_callbacks(self) -> List[keras.callbacks.Callback]:
        """Get training callbacks based on configuration."""
        callbacks = []
        
        # Model checkpoint
        if self.config.save_best_only:
            checkpoint_path = os.path.join(
                self.config.model_save_path,
                f"best_{self.config.architecture}_model.h5"
            )
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                )
            )
        
        # Early stopping
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        )
        
        # Reduce learning rate on plateau
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.early_stopping_patience // 2,
                min_lr=1e-7,
                verbose=1
            )
        )
        
        # TensorBoard logging
        log_dir = os.path.join(
            self.config.log_dir,
            f"tensorboard_{self.config.architecture}_{int(time.time())}"
        )
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        )
        return callbacks
    
    def _get_training_callbacks_with_error_handling(self) -> List[keras.callbacks.Callback]:
        """Get training callbacks with comprehensive error handling."""
        callbacks = self._get_training_callbacks()
        
        # Add robust training callback for error handling
        robust_callback = RobustTrainingCallback(
            error_handler=self.error_handler,
            checkpoint_frequency=5,  # Save checkpoint every 5 epochs
            auto_recovery=True
        )
        callbacks.append(robust_callback)
        
        return callbacks
    
    def _calculate_performance_metrics(self,
                                     model: Union[LightweightCNN3D, EfficientLSTM, HybridCNNLSTM],
                                     val_dataset: tf.data.Dataset,
                                     training_time: float,
                                     training_history: TrainingHistory) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        self.logger.info("Calculating performance metrics...")
        
        # Get final validation accuracy
        final_accuracy = training_history.val_accuracy[-1] if training_history.val_accuracy else 0.0
        
        # Measure inference time
        inference_start = time.time()
        sample_batch = next(iter(val_dataset.take(1)))
        _ = model.model.predict(sample_batch[0], verbose=0)
        inference_time = (time.time() - inference_start) / len(sample_batch[0])
        
        # Calculate model size
        model_size = self._calculate_model_size(model.model)
        
        # For now, use simplified metrics
        # In a full implementation, this would calculate per-class precision, recall, F1
        performance_metrics = PerformanceMetrics(
            accuracy=final_accuracy,
            precision={'overall': final_accuracy},  # Simplified
            recall={'overall': final_accuracy},     # Simplified
            f1_score={'overall': final_accuracy},   # Simplified
            training_time=training_time,
            inference_time=inference_time,
            model_size=model_size
        )
        
        return performance_metrics
    
    def _calculate_model_size(self, model: keras.Model) -> float:
        """Calculate model size in MB."""
        param_count = model.count_params()
        # Estimate size: 4 bytes per float32 parameter
        size_bytes = param_count * 4
        if self.config.use_mixed_precision:
            size_bytes = param_count * 2  # 2 bytes per float16
        
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    
    def _save_model_with_metrics(self,
                               model: Union[LightweightCNN3D, EfficientLSTM, HybridCNNLSTM],
                               performance_metrics: PerformanceMetrics,
                               training_history: TrainingHistory) -> None:
        """
        Save model with performance metrics.
        
        Implements Requirements 2.3: Model saving with performance metrics
        """
        timestamp = int(time.time())
        
        # Save model
        model_filename = f"{self.config.architecture}_model_{timestamp}.h5"
        model_path = os.path.join(self.config.model_save_path, model_filename)
        model.save_model(model_path)
        
        # Save metrics
        metrics_filename = f"{self.config.architecture}_metrics_{timestamp}.json"
        metrics_path = os.path.join(self.config.model_save_path, metrics_filename)
        
        metrics_data = {
            'performance_metrics': asdict(performance_metrics),
            'training_history': asdict(training_history),
            'training_config': asdict(self.config),
            'model_path': model_path,
            'timestamp': timestamp
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        self.logger.info(f"Model saved to: {model_path}")
        self.logger.info(f"Metrics saved to: {metrics_path}")
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a previously trained model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Success status
        """
        try:
            if self.config.architecture == '3dcnn':
                self.model = LightweightCNN3D()
                self.model.load_model(model_path)
            elif self.config.architecture == 'lstm':
                self.model = EfficientLSTM()
                self.model.load_model(model_path)
            elif self.config.architecture == 'hybrid':
                self.model = HybridCNNLSTM()
                self.model.load_model(model_path)
            else:
                raise ValueError(f"Unsupported architecture: {self.config.architecture}")
            
            self.logger.info(f"Model loaded successfully from: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results."""
        if not self.training_history or not self.performance_metrics:
            return {"status": "No training completed"}
        
        return {
            "architecture": self.config.architecture,
            "final_accuracy": self.performance_metrics.accuracy,
            "training_time": self.performance_metrics.training_time,
            "inference_time": self.performance_metrics.inference_time,
            "model_size_mb": self.performance_metrics.model_size,
            "epochs_trained": self.training_history.epochs,
            "best_epoch": self.training_history.best_epoch
        }


def compare_architectures(datasets: Dataset,
                         architectures: List[str] = ['3dcnn', 'lstm', 'hybrid'],
                         base_config: Optional[TrainingConfig] = None) -> List[ModelComparison]:
    """
    Compare multiple model architectures on the same dataset.
    
    Args:
        datasets: Dataset to train on
        architectures: List of architectures to compare
        base_config: Base configuration to use for all models
        
    Returns:
        List of model comparison results
    """
    results = []
    
    for arch in architectures:
        print(f"\nTraining {arch} architecture...")
        
        # Create config for this architecture
        config = base_config or TrainingConfig()
        config.architecture = arch
        
        # Train model
        trainer = ModelTrainer(config)
        success, history, metrics = trainer.train_model(datasets)
        
        if success:
            # Create comparison result
            comparison = ModelComparison(
                architecture_name=arch,
                performance_metrics=metrics,
                training_history=history,
                model_path=f"models/{arch}_model_{int(time.time())}.h5",
                hyperparameters=asdict(config)
            )
            results.append(comparison)
            
            print(f"{arch} training completed - Accuracy: {metrics.accuracy:.4f}")
        else:
            print(f"{arch} training failed")
    
    return results