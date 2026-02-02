"""
Additional methods for ModelTrainer class.
This file contains the error handling and recovery methods.
"""

from typing import Dict, List, Any, Tuple
from pathlib import Path
from dataclasses import asdict
import tensorflow as tf


def get_error_summary(self) -> Dict[str, Any]:
    """
    Get comprehensive error summary for debugging.
    
    Implements Requirements 2.4: Detailed error messages for debugging
    
    Returns:
        Dictionary with error summary and debugging information
    """
    error_summary = self.error_handler.get_error_summary()
    
    # Add training-specific context
    training_summary = {
        "training_config": asdict(self.config),
        "model_architecture": self.config.architecture,
        "training_completed": self.training_history is not None,
        "performance_metrics": asdict(self.performance_metrics) if self.performance_metrics else None
    }
    
    return {
        "error_summary": error_summary,
        "training_context": training_summary,
        "available_checkpoints": len(self.error_handler.checkpoints),
        "recovery_options": self._get_recovery_options()
    }


def _get_recovery_options(self) -> List[str]:
    """Get available recovery options based on current state."""
    options = []
    
    # Check for checkpoints
    if self.error_handler.checkpoints:
        options.append("Resume training from latest checkpoint")
        options.append("Resume training from specific checkpoint")
    
    # Check for saved models
    model_dir = Path(self.config.model_save_path)
    if model_dir.exists():
        model_files = list(model_dir.glob(f"{self.config.architecture}_model_*.h5"))
        if model_files:
            options.append("Load previously trained model")
    
    # General recovery options
    options.extend([
        "Restart training with reduced batch size",
        "Restart training with lower learning rate",
        "Switch to different model architecture",
        "Validate dataset and fix issues",
        "Clear GPU memory and restart"
    ])
    
    return options


def recover_from_error(self, recovery_strategy: str = "auto") -> bool:
    """
    Attempt to recover from training errors.
    
    Args:
        recovery_strategy: Recovery strategy ('auto', 'checkpoint', 'reduce_batch', etc.)
        
    Returns:
        Success status of recovery attempt
    """
    try:
        self.logger.info(f"Attempting recovery with strategy: {recovery_strategy}")
        
        if recovery_strategy == "auto":
            # Try automatic recovery based on error type
            if self.error_handler.errors:
                last_error = self.error_handler.errors[-1]
                if "memory" in last_error.error_message.lower():
                    return self._recover_from_memory_error()
                elif "nan" in last_error.error_message.lower():
                    return self._recover_from_nan_error()
                else:
                    return self._recover_from_checkpoint()
            
        elif recovery_strategy == "checkpoint":
            return self._recover_from_checkpoint()
            
        elif recovery_strategy == "reduce_batch":
            return self._recover_from_memory_error()
            
        elif recovery_strategy == "reduce_lr":
            return self._recover_from_nan_error()
            
        else:
            self.logger.warning(f"Unknown recovery strategy: {recovery_strategy}")
            return False
            
    except Exception as e:
        self.logger.error(f"Recovery attempt failed: {str(e)}")
        return False


def _recover_from_checkpoint(self) -> bool:
    """Recover by resuming from latest checkpoint."""
    checkpoint = self.error_handler.load_latest_checkpoint()
    if not checkpoint:
        self.logger.warning("No checkpoint available for recovery")
        return False
    
    try:
        # Update config to resume from checkpoint
        self.config.epochs = max(self.config.epochs - checkpoint.epoch, 10)
        self.logger.info(f"Recovery: Resuming from epoch {checkpoint.epoch}")
        return True
        
    except Exception as e:
        self.logger.error(f"Checkpoint recovery failed: {str(e)}")
        return False


def _recover_from_memory_error(self) -> bool:
    """Recover from out-of-memory errors."""
    try:
        # Reduce batch size
        original_batch_size = self.config.batch_size
        self.config.batch_size = max(1, self.config.batch_size // 2)
        
        # Enable mixed precision if not already enabled
        if not self.config.use_mixed_precision:
            self.config.use_mixed_precision = True
            self.logger.info("Recovery: Enabled mixed precision training")
        
        self.logger.info(f"Recovery: Reduced batch size from {original_batch_size} to {self.config.batch_size}")
        
        # Clear GPU memory
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()
            self.logger.info("Recovery: Cleared GPU memory")
        
        return True
        
    except Exception as e:
        self.logger.error(f"Memory error recovery failed: {str(e)}")
        return False


def _recover_from_nan_error(self) -> bool:
    """Recover from NaN/Inf loss errors."""
    try:
        # Reduce learning rate
        original_lr = self.config.learning_rate
        self.config.learning_rate = original_lr * 0.1
        
        self.logger.info(f"Recovery: Reduced learning rate from {original_lr} to {self.config.learning_rate}")
        
        # Add gradient clipping (would need to modify model compilation)
        self.logger.info("Recovery: Consider adding gradient clipping to model")
        
        return True
        
    except Exception as e:
        self.logger.error(f"NaN error recovery failed: {str(e)}")
        return False


def validate_training_environment(self) -> Tuple[bool, List[str]]:
    """
    Validate training environment and configuration.
    
    Returns:
        Tuple of (is_valid, issues_found)
    """
    issues = []
    
    try:
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            issues.append("No GPU detected - training will be slow on CPU")
        else:
            self.logger.info(f"Found {len(gpus)} GPU(s)")
            
            # Check GPU memory
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                issues.append(f"Could not configure GPU memory growth: {str(e)}")
        
        # Check disk space
        model_dir = Path(self.config.model_save_path)
        try:
            import shutil
            free_space = shutil.disk_usage(model_dir.parent).free / (1024**3)  # GB
            if free_space < 1.0:
                issues.append(f"Low disk space: {free_space:.1f}GB available")
        except Exception as e:
            issues.append(f"Could not check disk space: {str(e)}")
        
        # Validate configuration
        if self.config.batch_size < 1:
            issues.append("Invalid batch size: must be >= 1")
        
        if self.config.learning_rate <= 0:
            issues.append("Invalid learning rate: must be > 0")
        
        if self.config.epochs < 1:
            issues.append("Invalid epochs: must be >= 1")
        
        # Check TensorFlow version compatibility
        tf_version = tf.__version__
        self.logger.info(f"TensorFlow version: {tf_version}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
        
    except Exception as e:
        issues.append(f"Environment validation failed: {str(e)}")
        return False, issues


def cleanup_training_artifacts(self, keep_best_model: bool = True) -> None:
    """
    Clean up training artifacts to free disk space.
    
    Args:
        keep_best_model: Whether to keep the best performing model
    """
    try:
        self.logger.info("Cleaning up training artifacts...")
        
        # Clean up old checkpoints (keep only latest)
        if len(self.error_handler.checkpoints) > 1:
            self.error_handler.checkpoints = self.error_handler.checkpoints[-1:]
            self.error_handler._cleanup_old_checkpoints()
        
        # Clean up old log files (keep only recent ones)
        log_dir = Path(self.config.log_dir)
        if log_dir.exists():
            log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime)
            if len(log_files) > 5:  # Keep only 5 most recent log files
                for old_log in log_files[:-5]:
                    try:
                        old_log.unlink()
                        self.logger.info(f"Removed old log file: {old_log.name}")
                    except Exception as e:
                        self.logger.warning(f"Could not remove log file {old_log}: {str(e)}")
        
        # Clean up temporary model files if keeping best model
        if keep_best_model:
            model_dir = Path(self.config.model_save_path)
            if model_dir.exists():
                model_files = list(model_dir.glob(f"{self.config.architecture}_model_*.h5"))
                if len(model_files) > 3:  # Keep only 3 most recent models
                    model_files.sort(key=lambda x: x.stat().st_mtime)
                    for old_model in model_files[:-3]:
                        try:
                            old_model.unlink()
                            # Also remove corresponding metrics file
                            metrics_file = old_model.with_suffix('.json')
                            if metrics_file.exists():
                                metrics_file.unlink()
                            self.logger.info(f"Removed old model: {old_model.name}")
                        except Exception as e:
                            self.logger.warning(f"Could not remove model file {old_model}: {str(e)}")
        
        self.logger.info("Training artifacts cleanup completed")
        
    except Exception as e:
        self.logger.error(f"Cleanup failed: {str(e)}")


# Monkey patch the methods to ModelTrainer class
def add_error_handling_methods():
    """Add error handling methods to ModelTrainer class."""
    from .model_trainer import ModelTrainer
    
    ModelTrainer.get_error_summary = get_error_summary
    ModelTrainer._get_recovery_options = _get_recovery_options
    ModelTrainer.recover_from_error = recover_from_error
    ModelTrainer._recover_from_checkpoint = _recover_from_checkpoint
    ModelTrainer._recover_from_memory_error = _recover_from_memory_error
    ModelTrainer._recover_from_nan_error = _recover_from_nan_error
    ModelTrainer.validate_training_environment = validate_training_environment
    ModelTrainer.cleanup_training_artifacts = cleanup_training_artifacts