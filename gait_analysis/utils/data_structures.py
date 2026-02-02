"""Core data structures for the gait analysis system."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


@dataclass
class ValidationResult:
    """Result of video validation."""
    is_valid: bool
    resolution: Tuple[int, int]
    duration: float
    format: str
    error_message: Optional[str] = None


@dataclass
class PoseKeypoint:
    """3D pose keypoint with confidence."""
    x: float
    y: float
    z: float
    confidence: float


@dataclass
class PoseSequence:
    """Sequence of pose keypoints over time."""
    keypoints: List[List[PoseKeypoint]]  # [frame][keypoint]
    timestamps: List[float]
    confidence_scores: List[float]


@dataclass
class TrackedPose:
    """Pose sequence with tracking information."""
    pose_sequence: PoseSequence
    tracking_id: int
    tracking_confidence: float


@dataclass
class GaitParameters:
    """Calculated gait parameters."""
    stride_length: float
    cadence: float
    step_width: float
    swing_time: float
    stance_time: float
    double_support_time: float


@dataclass
class AsymmetryMetrics:
    """Gait asymmetry measurements."""
    left_right_stride_ratio: float
    left_right_swing_ratio: float
    left_right_stance_ratio: float
    temporal_asymmetry: float


@dataclass
class ClassificationResult:
    """Result of gait abnormality classification."""
    abnormality_type: str
    confidence: float
    severity_score: float
    affected_limbs: List[str]


@dataclass
class ClinicalInsights:
    """Clinical insights and recommendations."""
    primary_abnormalities: List[str]
    gait_parameters: GaitParameters
    asymmetry_metrics: AsymmetryMetrics
    recommendations: List[str]
    risk_factors: List[str]


@dataclass
class PerformanceMetrics:
    """Model performance metrics."""
    accuracy: float
    precision: Dict[str, float]  # per-class precision
    recall: Dict[str, float]     # per-class recall
    f1_score: Dict[str, float]   # per-class F1
    training_time: float         # seconds
    inference_time: float        # seconds per sample
    model_size: float           # MB


@dataclass
class TrainingHistory:
    """Training history and metrics."""
    train_loss: List[float]
    val_loss: List[float]
    train_accuracy: List[float]
    val_accuracy: List[float]
    epochs: int
    best_epoch: int


@dataclass
class ModelComparison:
    """Comparison between different model architectures."""
    architecture_name: str
    performance_metrics: PerformanceMetrics
    training_history: TrainingHistory
    model_path: str
    hyperparameters: Dict[str, Any]


@dataclass
class TrainingExample:
    """Single training example."""
    video_path: str
    pose_sequence: PoseSequence
    ground_truth_label: str
    severity_score: float
    metadata: Dict[str, Any]


@dataclass
class Dataset:
    """Training/validation dataset."""
    examples: List[TrainingExample]
    class_distribution: Dict[str, int]
    validation_split: float


@dataclass
class GaitCycle:
    """Single gait cycle data."""
    start_frame: int
    end_frame: int
    features: np.ndarray
    cycle_time: float
    step_count: int