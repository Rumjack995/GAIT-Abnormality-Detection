"""
Comprehensive error handling and logging for training failures.

This module implements detailed error logging, partial progress preservation,
and recovery mechanisms as specified in Requirements 2.4.
"""

import os
import json
import pickle
import logging
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import tensorflow as tf
from tensorflow import keras


@dataclass
class TrainingError:
    """Detailed training error information."""
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: str
    epoch: Optional[int] = None
    batch: Optional[int] = None
    model_state: Optional[str] = None
    recovery_suggestions: List[str] = None


@dataclass
class TrainingCheckpoint:
    """Training checkpoint for progress preservation."""
    epoch: int
    batch: int
    model_weights_path: str
    optimizer_state_path: str
    training_history: Dict[str, List[float]]
    timestamp: str
    config: Dict[str, Any]


class TrainingErrorHandler:
    """
    Comprehensive error handler for training failures.
    
    Implements Requirements 2.4: Error logging and partial progress preservation
    """
    
    def __init__(self, 
                 log_dir: str = "logs/",
                 checkpoint_dir: str = "checkpoints/",
                 max_checkpoints: int = 5):
        """
        Initialize error handler.
        
        Args:
            log_dir: Directory for error logs
            checkpoint_dir: Directory for training checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_error_logging()
        
        # Error tracking
        self.errors = []
        self.checkpoints = []
        
    def _setup_error_logging(self) -> logging.Logger:
        """Setup detailed error logging."""
        logger = logging.getLogger("training_error_handler")
        logger.setLevel(logging.DEBUG)
        
        # Create error log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_log_file = self.log_dir / f"training_errors_{timestamp}.log"
        
        # File handler for errors
        file_handler = logging.FileHandler(error_log_file)
        file_handler.setLevel(logging.ERROR)
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Detailed formatter for error logs
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(detailed_formatter)
        
        # Simple formatter for console
        simple_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers if not already added
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def log_training_error(self,
                          error: Exception,
                          context: Dict[str, Any],
                          epoch: Optional[int] = None,
                          batch: Optional[int] = None) -> TrainingError:
        """
        Log detailed training error with context.
        
        Args:
            error: Exception that occurred
            context: Training context (model, config, etc.)
            epoch: Current epoch when error occurred
            batch: Current batch when error occurred
            
        Returns:
            TrainingError object with detailed information
        """
        # Create detailed error record
        error_record = TrainingError(
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            timestamp=datetime.now().isoformat(),
            epoch=epoch,
            batch=batch,
            model_state=self._get_model_state_summary(context.get('model')),
            recovery_suggestions=self._generate_recovery_suggestions(error, context)
        )
        
        # Log error details
        self.logger.error(f"Training Error Occurred:")
        self.logger.error(f"  Type: {error_record.error_type}")
        self.logger.error(f"  Message: {error_record.error_message}")
        self.logger.error(f"  Epoch: {error_record.epoch}")
        self.logger.error(f"  Batch: {error_record.batch}")
        self.logger.error(f"  Stack Trace:\n{error_record.stack_trace}")
        
        # Log recovery suggestions
        if error_record.recovery_suggestions:
            self.logger.error("Recovery Suggestions:")
            for suggestion in error_record.recovery_suggestions:
                self.logger.error(f"  - {suggestion}")
        
        # Save error to file
        self._save_error_record(error_record)
        
        # Add to error list
        self.errors.append(error_record)
        
        return error_record
    
    def _get_model_state_summary(self, model) -> Optional[str]:
        """Get summary of model state for debugging."""
        if model is None:
            return "Model is None"
        
        try:
            if hasattr(model, 'model') and model.model is not None:
                keras_model = model.model
                return f"Model compiled: {keras_model.built}, Params: {keras_model.count_params():,}"
            else:
                return "Model not built or accessible"
        except Exception as e:
            return f"Error getting model state: {str(e)}"
    
    def _generate_recovery_suggestions(self, 
                                     error: Exception, 
                                     context: Dict[str, Any]) -> List[str]:
        """Generate specific recovery suggestions based on error type."""
        suggestions = []
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Memory-related errors
        if "out of memory" in error_msg or "oom" in error_msg:
            suggestions.extend([
                "Reduce batch size in training configuration",
                "Enable mixed precision training (FP16)",
                "Use gradient accumulation for effective larger batch sizes",
                "Clear GPU memory: tf.keras.backend.clear_session()",
                "Check for memory leaks in data pipeline"
            ])
        
        # Data-related errors
        elif "invalid argument" in error_msg or "shape" in error_msg:
            suggestions.extend([
                "Verify input data shapes match model expectations",
                "Check dataset preprocessing pipeline",
                "Validate batch dimensions and data types",
                "Ensure consistent sequence lengths in LSTM inputs"
            ])
        
        # Model compilation errors
        elif "not compiled" in error_msg or "optimizer" in error_msg:
            suggestions.extend([
                "Ensure model is compiled before training",
                "Check optimizer configuration and learning rate",
                "Verify loss function matches output layer activation",
                "Rebuild and recompile the model"
            ])
        
        # File I/O errors
        elif "file" in error_msg or "directory" in error_msg:
            suggestions.extend([
                "Check file paths and permissions",
                "Ensure sufficient disk space for model saving",
                "Verify checkpoint directory exists and is writable",
                "Check for corrupted checkpoint files"
            ])
        
        # CUDA/GPU errors
        elif "cuda" in error_msg or "gpu" in error_msg:
            suggestions.extend([
                "Check GPU availability and CUDA installation",
                "Restart training session to reset GPU state",
                "Try training on CPU as fallback",
                "Update GPU drivers and TensorFlow version"
            ])
        
        # Convergence issues
        elif "nan" in error_msg or "inf" in error_msg:
            suggestions.extend([
                "Reduce learning rate to prevent gradient explosion",
                "Add gradient clipping to training configuration",
                "Check for numerical instability in loss function",
                "Use batch normalization or layer normalization",
                "Verify input data normalization"
            ])
        
        # General suggestions
        suggestions.extend([
            "Check training logs for patterns before failure",
            "Try resuming from last checkpoint if available",
            "Validate dataset integrity and format",
            "Consider reducing model complexity for debugging"
        ])
        
        return suggestions
    
    def _save_error_record(self, error_record: TrainingError) -> None:
        """Save error record to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = self.log_dir / f"error_record_{timestamp}.json"
        
        try:
            with open(error_file, 'w') as f:
                json.dump(asdict(error_record), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save error record: {str(e)}")
    
    def save_training_checkpoint(self,
                               model,
                               optimizer,
                               epoch: int,
                               batch: int,
                               training_history: Dict[str, List[float]],
                               config: Dict[str, Any]) -> Optional[TrainingCheckpoint]:
        """
        Save training checkpoint for progress preservation.
        
        Args:
            model: Current model instance
            optimizer: Current optimizer state
            epoch: Current epoch
            batch: Current batch
            training_history: Training history so far
            config: Training configuration
            
        Returns:
            TrainingCheckpoint object if successful, None otherwise
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_id = f"checkpoint_epoch_{epoch}_batch_{batch}_{timestamp}"
            
            # Create checkpoint directory
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            checkpoint_path.mkdir(exist_ok=True)
            
            # Save model weights
            model_weights_path = checkpoint_path / "model_weights.h5"
            if hasattr(model, 'model') and model.model is not None:
                model.model.save_weights(str(model_weights_path))
            else:
                self.logger.warning("Model not available for checkpoint saving")
                return None
            
            # Save optimizer state
            optimizer_state_path = checkpoint_path / "optimizer_state.pkl"
            try:
                optimizer_weights = optimizer.get_weights()
                with open(optimizer_state_path, 'wb') as f:
                    pickle.dump(optimizer_weights, f)
            except Exception as e:
                self.logger.warning(f"Could not save optimizer state: {str(e)}")
                optimizer_state_path = None
            
            # Create checkpoint record
            checkpoint = TrainingCheckpoint(
                epoch=epoch,
                batch=batch,
                model_weights_path=str(model_weights_path),
                optimizer_state_path=str(optimizer_state_path) if optimizer_state_path else None,
                training_history=training_history.copy(),
                timestamp=timestamp,
                config=config.copy()
            )
            
            # Save checkpoint metadata
            checkpoint_file = checkpoint_path / "checkpoint_info.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(asdict(checkpoint), f, indent=2)
            
            # Add to checkpoint list
            self.checkpoints.append(checkpoint)
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            self.logger.info(f"Training checkpoint saved: {checkpoint_id}")
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Failed to save training checkpoint: {str(e)}")
            return None
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to save disk space."""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by timestamp and remove oldest
            self.checkpoints.sort(key=lambda x: x.timestamp)
            
            while len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                
                # Remove checkpoint directory
                try:
                    checkpoint_path = Path(old_checkpoint.model_weights_path).parent
                    if checkpoint_path.exists():
                        import shutil
                        shutil.rmtree(checkpoint_path)
                    self.logger.info(f"Removed old checkpoint: {checkpoint_path.name}")
                except Exception as e:
                    self.logger.warning(f"Could not remove old checkpoint: {str(e)}")
    
    def load_latest_checkpoint(self) -> Optional[TrainingCheckpoint]:
        """
        Load the most recent training checkpoint.
        
        Returns:
            Latest TrainingCheckpoint if available, None otherwise
        """
        if not self.checkpoints:
            # Try to find checkpoints in directory
            self._discover_existing_checkpoints()
        
        if not self.checkpoints:
            self.logger.info("No training checkpoints found")
            return None
        
        # Return most recent checkpoint
        latest_checkpoint = max(self.checkpoints, key=lambda x: x.timestamp)
        self.logger.info(f"Latest checkpoint found: epoch {latest_checkpoint.epoch}")
        return latest_checkpoint
    
    def _discover_existing_checkpoints(self) -> None:
        """Discover existing checkpoints in checkpoint directory."""
        try:
            for checkpoint_dir in self.checkpoint_dir.iterdir():
                if checkpoint_dir.is_dir():
                    checkpoint_file = checkpoint_dir / "checkpoint_info.json"
                    if checkpoint_file.exists():
                        try:
                            with open(checkpoint_file, 'r') as f:
                                checkpoint_data = json.load(f)
                            
                            checkpoint = TrainingCheckpoint(**checkpoint_data)
                            self.checkpoints.append(checkpoint)
                        except Exception as e:
                            self.logger.warning(f"Could not load checkpoint {checkpoint_dir}: {str(e)}")
        except Exception as e:
            self.logger.warning(f"Error discovering checkpoints: {str(e)}")
    
    def resume_training_from_checkpoint(self,
                                      model,
                                      optimizer,
                                      checkpoint: TrainingCheckpoint) -> bool:
        """
        Resume training from a saved checkpoint.
        
        Args:
            model: Model instance to restore
            optimizer: Optimizer instance to restore
            checkpoint: Checkpoint to restore from
            
        Returns:
            Success status
        """
        try:
            self.logger.info(f"Resuming training from epoch {checkpoint.epoch}, batch {checkpoint.batch}")
            
            # Load model weights
            if hasattr(model, 'model') and model.model is not None:
                model.model.load_weights(checkpoint.model_weights_path)
                self.logger.info("Model weights restored successfully")
            else:
                self.logger.error("Model not available for weight restoration")
                return False
            
            # Load optimizer state if available
            if checkpoint.optimizer_state_path and os.path.exists(checkpoint.optimizer_state_path):
                try:
                    with open(checkpoint.optimizer_state_path, 'rb') as f:
                        optimizer_weights = pickle.load(f)
                    optimizer.set_weights(optimizer_weights)
                    self.logger.info("Optimizer state restored successfully")
                except Exception as e:
                    self.logger.warning(f"Could not restore optimizer state: {str(e)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to resume from checkpoint: {str(e)}")
            return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all training errors."""
        if not self.errors:
            return {"total_errors": 0, "message": "No training errors recorded"}
        
        error_types = {}
        for error in self.errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        return {
            "total_errors": len(self.errors),
            "error_types": error_types,
            "latest_error": {
                "type": self.errors[-1].error_type,
                "message": self.errors[-1].error_message,
                "timestamp": self.errors[-1].timestamp
            },
            "recovery_suggestions": self.errors[-1].recovery_suggestions if self.errors else []
        }
    
    def clear_error_history(self) -> None:
        """Clear error history (useful for new training sessions)."""
        self.errors.clear()
        self.logger.info("Error history cleared")


class RobustTrainingCallback(keras.callbacks.Callback):
    """
    Keras callback for robust training with automatic error handling.
    """
    
    def __init__(self, 
                 error_handler: TrainingErrorHandler,
                 checkpoint_frequency: int = 10,
                 auto_recovery: bool = True):
        """
        Initialize robust training callback.
        
        Args:
            error_handler: Error handler instance
            checkpoint_frequency: Save checkpoint every N epochs
            auto_recovery: Attempt automatic recovery on errors
        """
        super().__init__()
        self.error_handler = error_handler
        self.checkpoint_frequency = checkpoint_frequency
        self.auto_recovery = auto_recovery
        self.current_epoch = 0
        self.current_batch = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch."""
        self.current_epoch = epoch
        self.current_batch = 0
    
    def on_batch_begin(self, batch, logs=None):
        """Called at the beginning of each batch."""
        self.current_batch = batch
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        # Save checkpoint periodically
        if (epoch + 1) % self.checkpoint_frequency == 0:
            try:
                training_history = {
                    'loss': getattr(self.model.history, 'history', {}).get('loss', []),
                    'val_loss': getattr(self.model.history, 'history', {}).get('val_loss', []),
                    'accuracy': getattr(self.model.history, 'history', {}).get('accuracy', []),
                    'val_accuracy': getattr(self.model.history, 'history', {}).get('val_accuracy', [])
                }
                
                self.error_handler.save_training_checkpoint(
                    model=self.model,
                    optimizer=self.model.optimizer,
                    epoch=epoch,
                    batch=self.current_batch,
                    training_history=training_history,
                    config={}  # Would include actual config in real implementation
                )
            except Exception as e:
                self.error_handler.logger.warning(f"Could not save checkpoint: {str(e)}")
    
    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of each training batch."""
        # Check for NaN losses
        if logs and 'loss' in logs:
            if np.isnan(logs['loss']) or np.isinf(logs['loss']):
                error_msg = f"NaN or Inf loss detected at epoch {self.current_epoch}, batch {batch}"
                error = ValueError(error_msg)
                
                context = {
                    'model': self.model,
                    'logs': logs,
                    'epoch': self.current_epoch,
                    'batch': batch
                }
                
                self.error_handler.log_training_error(
                    error=error,
                    context=context,
                    epoch=self.current_epoch,
                    batch=batch
                )
                
                # Stop training on NaN loss
                self.model.stop_training = True