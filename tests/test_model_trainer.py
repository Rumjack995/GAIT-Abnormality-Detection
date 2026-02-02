"""
Tests for ModelTrainer class with error handling and logging.

This module tests the training system with dataset validation,
error handling, and performance tracking.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch

from gait_analysis.training import (
    ModelTrainer, 
    TrainingConfig, 
    DatasetValidator,
    TrainingErrorHandler
)
from gait_analysis.utils.data_structures import (
    Dataset, 
    TrainingExample, 
    PoseSequence,
    PoseKeypoint
)


class TestDatasetValidator:
    """Test cases for dataset validation."""
    
    def test_empty_dataset_validation(self):
        """Test validation of empty dataset."""
        validator = DatasetValidator()
        
        empty_dataset = Dataset(
            examples=[],
            class_distribution={},
            validation_split=0.2
        )
        
        is_valid, errors = validator.validate_dataset(empty_dataset)
        
        assert not is_valid
        assert "Dataset is empty" in str(errors)
    
    def test_valid_dataset_validation(self):
        """Test validation of properly formatted dataset."""
        validator = DatasetValidator()
        
        # Create mock pose sequence
        pose_keypoints = [
            [PoseKeypoint(x=0.5, y=0.5, z=0.0, confidence=0.9) for _ in range(33)]
            for _ in range(10)  # 10 frames
        ]
        
        pose_sequence = PoseSequence(
            keypoints=pose_keypoints,
            timestamps=[i * 0.1 for i in range(10)],
            confidence_scores=[0.9 for _ in range(10)]
        )
        
        # Create valid training examples
        examples = []
        for i in range(10):
            example = TrainingExample(
                video_path=f"test_video_{i}.mp4",
                pose_sequence=pose_sequence,
                ground_truth_label="normal" if i < 5 else "abnormal",
                severity_score=0.1 if i < 5 else 0.8,
                metadata={"test": True}
            )
            examples.append(example)
        
        dataset = Dataset(
            examples=examples,
            class_distribution={"normal": 5, "abnormal": 5},
            validation_split=0.2
        )
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            is_valid, errors = validator.validate_dataset(dataset)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_validation_split(self):
        """Test validation with invalid validation split."""
        validator = DatasetValidator()
        
        dataset = Dataset(
            examples=[Mock()],  # Mock example
            class_distribution={"test": 1},
            validation_split=1.5  # Invalid split
        )
        
        is_valid, errors = validator.validate_dataset(dataset)
        
        assert not is_valid
        assert any("validation split" in error.lower() for error in errors)


class TestTrainingErrorHandler:
    """Test cases for training error handler."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, "logs")
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        
        self.error_handler = TrainingErrorHandler(
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            max_checkpoints=3
        )
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_error_logging(self):
        """Test error logging functionality."""
        test_error = ValueError("Test error message")
        context = {"model": None, "config": {"test": True}}
        
        error_record = self.error_handler.log_training_error(
            error=test_error,
            context=context,
            epoch=5,
            batch=10
        )
        
        assert error_record.error_type == "ValueError"
        assert error_record.error_message == "Test error message"
        assert error_record.epoch == 5
        assert error_record.batch == 10
        assert len(error_record.recovery_suggestions) > 0
    
    def test_recovery_suggestions_memory_error(self):
        """Test recovery suggestions for memory errors."""
        memory_error = RuntimeError("CUDA out of memory")
        context = {"model": None}
        
        error_record = self.error_handler.log_training_error(
            error=memory_error,
            context=context
        )
        
        suggestions = error_record.recovery_suggestions
        assert any("batch size" in suggestion.lower() for suggestion in suggestions)
        assert any("mixed precision" in suggestion.lower() for suggestion in suggestions)
    
    def test_checkpoint_saving(self):
        """Test checkpoint saving functionality."""
        # Mock model and optimizer
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.save_weights = Mock()
        
        mock_optimizer = Mock()
        mock_optimizer.get_weights = Mock(return_value=[np.array([1, 2, 3])])
        
        training_history = {"loss": [0.5, 0.3], "accuracy": [0.8, 0.9]}
        config = {"batch_size": 4, "learning_rate": 0.001}
        
        checkpoint = self.error_handler.save_training_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            epoch=10,
            batch=5,
            training_history=training_history,
            config=config
        )
        
        assert checkpoint is not None
        assert checkpoint.epoch == 10
        assert checkpoint.batch == 5
        assert checkpoint.training_history == training_history
        assert checkpoint.config == config
    
    def test_error_summary(self):
        """Test error summary generation."""
        # Log a few errors
        self.error_handler.log_training_error(
            ValueError("Error 1"), {"test": 1}
        )
        self.error_handler.log_training_error(
            RuntimeError("Error 2"), {"test": 2}
        )
        
        summary = self.error_handler.get_error_summary()
        
        assert summary["total_errors"] == 2
        assert "ValueError" in summary["error_types"]
        assert "RuntimeError" in summary["error_types"]
        assert "latest_error" in summary


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.config = TrainingConfig(
            architecture='lstm',  # Use LSTM for faster testing
            epochs=2,  # Minimal epochs for testing
            batch_size=2,
            model_save_path=os.path.join(self.temp_dir, "models"),
            log_dir=os.path.join(self.temp_dir, "logs"),
            use_mixed_precision=False  # Disable for testing
        )
        
        self.trainer = ModelTrainer(self.config)
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.config.architecture == 'lstm'
        assert self.trainer.error_handler is not None
        assert self.trainer.validator is not None
        assert self.trainer.logger is not None
    
    def test_environment_validation(self):
        """Test training environment validation."""
        is_valid, issues = self.trainer.validate_training_environment()
        
        # Should be valid or have minor issues
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)
    
    def test_error_summary(self):
        """Test error summary generation."""
        summary = self.trainer.get_error_summary()
        
        assert "error_summary" in summary
        assert "training_context" in summary
        assert "recovery_options" in summary
        assert isinstance(summary["recovery_options"], list)
    
    def test_recovery_options(self):
        """Test recovery options generation."""
        options = self.trainer._get_recovery_options()
        
        assert isinstance(options, list)
        assert len(options) > 0
        assert any("batch size" in option.lower() for option in options)
    
    @patch('gait_analysis.training.model_trainer.create_efficient_lstm')
    def test_model_creation(self, mock_create_lstm):
        """Test model creation."""
        # Mock the model creation
        mock_model = Mock()
        mock_model.compile_model = Mock()
        mock_create_lstm.return_value = mock_model
        
        model = self.trainer.create_model(num_classes=3)
        
        assert model is not None
        mock_create_lstm.assert_called_once()
        mock_model.compile_model.assert_called_once()
    
    def test_cleanup_artifacts(self):
        """Test training artifacts cleanup."""
        # Create some dummy files
        os.makedirs(self.config.model_save_path, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Create dummy model files
        for i in range(5):
            dummy_model = Path(self.config.model_save_path) / f"lstm_model_{i}.h5"
            dummy_model.touch()
        
        # Create dummy log files
        for i in range(8):
            dummy_log = Path(self.config.log_dir) / f"training_{i}.log"
            dummy_log.touch()
        
        # Run cleanup
        self.trainer.cleanup_training_artifacts(keep_best_model=True)
        
        # Check that files were cleaned up
        model_files = list(Path(self.config.model_save_path).glob("*.h5"))
        log_files = list(Path(self.config.log_dir).glob("*.log"))
        
        assert len(model_files) <= 3  # Should keep only 3 models
        assert len(log_files) <= 5   # Should keep only 5 logs


if __name__ == "__main__":
    # Run basic tests
    test_validator = TestDatasetValidator()
    test_validator.test_empty_dataset_validation()
    test_validator.test_invalid_validation_split()
    
    test_error_handler = TestTrainingErrorHandler()
    test_error_handler.setup_method()
    test_error_handler.test_error_logging()
    test_error_handler.test_recovery_suggestions_memory_error()
    test_error_handler.teardown_method()
    
    test_trainer = TestModelTrainer()
    test_trainer.setup_method()
    test_trainer.test_trainer_initialization()
    test_trainer.test_environment_validation()
    test_trainer.test_error_summary()
    test_trainer.teardown_method()
    
    print("All basic model trainer tests passed!")