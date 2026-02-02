"""Training components for gait abnormality detection models."""

from .model_trainer import ModelTrainer, TrainingConfig, DatasetValidator
from .error_handling import TrainingErrorHandler, RobustTrainingCallback, TrainingError, TrainingCheckpoint

# Add error handling methods to ModelTrainer
from .model_trainer_methods import add_error_handling_methods
add_error_handling_methods()

__all__ = [
    'ModelTrainer',
    'TrainingConfig', 
    'DatasetValidator',
    'TrainingErrorHandler',
    'RobustTrainingCallback',
    'TrainingError',
    'TrainingCheckpoint'
]