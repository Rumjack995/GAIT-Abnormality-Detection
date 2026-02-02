"""Data augmentation techniques for gait analysis training data."""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import random
from copy import deepcopy
import logging

from ..utils.data_structures import (
    PoseSequence, TrackedPose, PoseKeypoint, TrainingExample
)


class DataAugmentation:
    """
    Data augmentation pipeline for gait analysis training datasets.
    
    Implements augmentation strategies for limited training data including
    rotation, scaling, and temporal augmentation methods as specified
    in Requirement 6.2.
    """
    
    def __init__(self, 
                 rotation_range: float = 15.0,
                 scale_range: Tuple[float, float] = (0.8, 1.2),
                 noise_std: float = 0.02,
                 temporal_stretch_range: Tuple[float, float] = (0.8, 1.2),
                 flip_probability: float = 0.5):
        """Initialize data augmentation pipeline."""
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.temporal_stretch_range = temporal_stretch_range
        self.flip_probability = flip_probability
        self.logger = logging.getLogger(__name__)
        
        # MediaPipe landmark pairs for horizontal flipping
        self.flip_pairs = [
            (11, 12), (13, 14), (15, 16), (23, 24),
            (25, 26), (27, 28), (29, 30), (31, 32)
        ]
    
    def augment_training_dataset(self, 
                               training_examples: List[TrainingExample],
                               augmentation_factor: int = 2) -> List[TrainingExample]:
        """Apply augmentation to training dataset."""
        if not training_examples:
            return []
        
        augmented_examples = list(training_examples)  # Keep originals
        
        # Generate augmented versions
        for example in training_examples:
            for aug_idx in range(augmentation_factor):
                try:
                    augmented_example = self._augment_single_example(example, aug_idx)
                    augmented_examples.append(augmented_example)
                except Exception as e:
                    self.logger.warning(f"Failed to augment example: {e}")
                    continue
        
        self.logger.info(f"Augmented dataset from {len(training_examples)} to {len(augmented_examples)} examples")
        return augmented_examples
    
    def _augment_single_example(self, example: TrainingExample, aug_idx: int) -> TrainingExample:
        """Apply augmentation to a single training example."""
        # Create deep copy to avoid modifying original
        augmented_example = deepcopy(example)
        
        # Set random seed for reproducible augmentation (fix seed range issue)
        seed_value = abs(hash(example.video_path) + aug_idx) % (2**32 - 1)
        random.seed(seed_value)
        np.random.seed(seed_value)
        
        # Apply different augmentation techniques
        pose_sequence = augmented_example.pose_sequence
        
        # 1. Spatial augmentations
        if random.random() < 0.7:  # 70% chance of spatial augmentation
            pose_sequence = self._apply_spatial_augmentation(pose_sequence)
        
        # 2. Temporal augmentations
        if random.random() < 0.5:  # 50% chance of temporal augmentation
            pose_sequence = self._apply_temporal_augmentation(pose_sequence)
        
        # 3. Noise augmentation
        if random.random() < 0.6:  # 60% chance of noise augmentation
            pose_sequence = self._apply_noise_augmentation(pose_sequence)
        
        # Update the augmented example
        augmented_example.pose_sequence = pose_sequence
        augmented_example.video_path = f"{example.video_path}_aug_{aug_idx}"
        
        return augmented_example
    
    def _apply_spatial_augmentation(self, pose_sequence: PoseSequence) -> PoseSequence:
        """Apply spatial augmentation techniques."""
        augmented_sequence = deepcopy(pose_sequence)
        
        # Choose augmentation type
        aug_type = random.choice(['rotation', 'scaling', 'horizontal_flip'])
        
        if aug_type == 'rotation':
            augmented_sequence = self._apply_rotation(augmented_sequence)
        elif aug_type == 'scaling':
            augmented_sequence = self._apply_scaling(augmented_sequence)
        elif aug_type == 'horizontal_flip':
            augmented_sequence = self._apply_horizontal_flip(augmented_sequence)
        
        return augmented_sequence
    
    def _apply_rotation(self, pose_sequence: PoseSequence) -> PoseSequence:
        """Apply rotation augmentation around the vertical axis."""
        # Generate random rotation angle
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        angle_rad = np.radians(angle)
        
        # Rotation matrix for 2D rotation (around z-axis)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Apply rotation to all keypoints
        for frame_idx, frame_keypoints in enumerate(pose_sequence.keypoints):
            for kp_idx, keypoint in enumerate(frame_keypoints):
                if keypoint.confidence > 0.1:  # Only rotate confident keypoints
                    # Apply 2D rotation to x,y coordinates
                    original_pos = np.array([keypoint.x, keypoint.y])
                    rotated_pos = rotation_matrix @ original_pos
                    
                    # Update keypoint coordinates
                    pose_sequence.keypoints[frame_idx][kp_idx].x = rotated_pos[0]
                    pose_sequence.keypoints[frame_idx][kp_idx].y = rotated_pos[1]
                    # Keep z coordinate unchanged
        
        return pose_sequence
    
    def _apply_scaling(self, pose_sequence: PoseSequence) -> PoseSequence:
        """Apply scaling augmentation to pose keypoints."""
        # Generate random scale factors
        scale_x = random.uniform(*self.scale_range)
        scale_y = random.uniform(*self.scale_range)
        scale_z = random.uniform(*self.scale_range)
        
        # Apply scaling to all keypoints
        for frame_idx, frame_keypoints in enumerate(pose_sequence.keypoints):
            for kp_idx, keypoint in enumerate(frame_keypoints):
                if keypoint.confidence > 0.1:  # Only scale confident keypoints
                    pose_sequence.keypoints[frame_idx][kp_idx].x *= scale_x
                    pose_sequence.keypoints[frame_idx][kp_idx].y *= scale_y
                    pose_sequence.keypoints[frame_idx][kp_idx].z *= scale_z
        
        return pose_sequence
    
    def _apply_horizontal_flip(self, pose_sequence: PoseSequence) -> PoseSequence:
        """Apply horizontal flip augmentation by swapping left/right body parts."""
        if random.random() > self.flip_probability:
            return pose_sequence
        
        # Apply horizontal flip to all frames
        for frame_idx, frame_keypoints in enumerate(pose_sequence.keypoints):
            # Flip x coordinates
            for kp_idx, keypoint in enumerate(frame_keypoints):
                if keypoint.confidence > 0.1:
                    pose_sequence.keypoints[frame_idx][kp_idx].x = -keypoint.x
            
            # Swap left/right landmark pairs
            for left_idx, right_idx in self.flip_pairs:
                if (left_idx < len(frame_keypoints) and right_idx < len(frame_keypoints)):
                    # Swap the keypoints
                    left_kp = deepcopy(frame_keypoints[left_idx])
                    right_kp = deepcopy(frame_keypoints[right_idx])
                    
                    pose_sequence.keypoints[frame_idx][left_idx] = right_kp
                    pose_sequence.keypoints[frame_idx][right_idx] = left_kp
        
        return pose_sequence
    
    def _apply_temporal_augmentation(self, pose_sequence: PoseSequence) -> PoseSequence:
        """Apply temporal augmentation including time stretching and frame sampling."""
        aug_type = random.choice(['time_stretch', 'frame_dropout', 'temporal_shift'])
        
        if aug_type == 'time_stretch':
            return self._apply_time_stretching(pose_sequence)
        elif aug_type == 'frame_dropout':
            return self._apply_frame_dropout(pose_sequence)
        elif aug_type == 'temporal_shift':
            return self._apply_temporal_shift(pose_sequence)
        
        return pose_sequence
    
    def _apply_time_stretching(self, pose_sequence: PoseSequence) -> PoseSequence:
        """Apply time stretching by resampling the temporal sequence."""
        stretch_factor = random.uniform(*self.temporal_stretch_range)
        original_length = len(pose_sequence.keypoints)
        new_length = int(original_length * stretch_factor)
        
        if new_length < 5:  # Minimum sequence length
            return pose_sequence
        
        # Create new indices for resampling
        new_indices = np.linspace(0, original_length - 1, new_length)
        
        # Resample keypoints
        new_keypoints = []
        new_timestamps = []
        new_confidence_scores = []
        
        for new_idx in new_indices:
            # Find closest original frame
            closest_idx = int(np.round(new_idx))
            closest_idx = max(0, min(closest_idx, original_length - 1))
            
            new_keypoints.append(deepcopy(pose_sequence.keypoints[closest_idx]))
            
            # Interpolate timestamps
            if closest_idx < len(pose_sequence.timestamps):
                new_timestamps.append(pose_sequence.timestamps[closest_idx])
            else:
                new_timestamps.append(pose_sequence.timestamps[-1])
            
            # Interpolate confidence scores
            if closest_idx < len(pose_sequence.confidence_scores):
                new_confidence_scores.append(pose_sequence.confidence_scores[closest_idx])
            else:
                new_confidence_scores.append(pose_sequence.confidence_scores[-1])
        
        return PoseSequence(
            keypoints=new_keypoints,
            timestamps=new_timestamps,
            confidence_scores=new_confidence_scores
        )
    
    def _apply_frame_dropout(self, pose_sequence: PoseSequence) -> PoseSequence:
        """Apply frame dropout by randomly removing frames."""
        dropout_rate = random.uniform(0.05, 0.15)  # Drop 5-15% of frames
        original_length = len(pose_sequence.keypoints)
        
        if original_length < 10:  # Don't dropout if sequence is too short
            return pose_sequence
        
        # Randomly select frames to keep
        keep_indices = []
        for i in range(original_length):
            if random.random() > dropout_rate:
                keep_indices.append(i)
        
        # Ensure minimum sequence length
        if len(keep_indices) < 5:
            keep_indices = list(range(0, original_length, max(1, original_length // 5)))
        
        # Create new sequence with selected frames
        new_keypoints = [pose_sequence.keypoints[i] for i in keep_indices]
        new_timestamps = [pose_sequence.timestamps[i] for i in keep_indices 
                         if i < len(pose_sequence.timestamps)]
        new_confidence_scores = [pose_sequence.confidence_scores[i] for i in keep_indices 
                               if i < len(pose_sequence.confidence_scores)]
        
        return PoseSequence(
            keypoints=new_keypoints,
            timestamps=new_timestamps,
            confidence_scores=new_confidence_scores
        )
    
    def _apply_temporal_shift(self, pose_sequence: PoseSequence) -> PoseSequence:
        """Apply temporal shift by starting the sequence at a different frame."""
        original_length = len(pose_sequence.keypoints)
        
        if original_length < 10:  # Don't shift if sequence is too short
            return pose_sequence
        
        # Random shift amount (up to 20% of sequence length)
        max_shift = int(original_length * 0.2)
        shift_amount = random.randint(1, max_shift)
        
        # Apply circular shift
        new_keypoints = (pose_sequence.keypoints[shift_amount:] + 
                        pose_sequence.keypoints[:shift_amount])
        
        new_timestamps = (pose_sequence.timestamps[shift_amount:] + 
                         pose_sequence.timestamps[:shift_amount])
        
        new_confidence_scores = (pose_sequence.confidence_scores[shift_amount:] + 
                               pose_sequence.confidence_scores[:shift_amount])
        
        return PoseSequence(
            keypoints=new_keypoints,
            timestamps=new_timestamps,
            confidence_scores=new_confidence_scores
        )
    
    def _apply_noise_augmentation(self, pose_sequence: PoseSequence) -> PoseSequence:
        """Apply Gaussian noise to pose keypoints."""
        # Apply noise to all keypoints
        for frame_idx, frame_keypoints in enumerate(pose_sequence.keypoints):
            for kp_idx, keypoint in enumerate(frame_keypoints):
                if keypoint.confidence > 0.1:  # Only add noise to confident keypoints
                    # Add Gaussian noise to coordinates
                    noise_x = np.random.normal(0, self.noise_std)
                    noise_y = np.random.normal(0, self.noise_std)
                    noise_z = np.random.normal(0, self.noise_std)
                    
                    pose_sequence.keypoints[frame_idx][kp_idx].x += noise_x
                    pose_sequence.keypoints[frame_idx][kp_idx].y += noise_y
                    pose_sequence.keypoints[frame_idx][kp_idx].z += noise_z
                    
                    # Slightly reduce confidence due to noise
                    confidence_reduction = np.random.uniform(0.0, 0.05)
                    new_confidence = max(0.0, keypoint.confidence - confidence_reduction)
                    pose_sequence.keypoints[frame_idx][kp_idx].confidence = new_confidence
        
        return pose_sequence
    
    def create_augmentation_pipeline(self, 
                                   dataset_size: int,
                                   target_size: int,
                                   augmentation_strategies: Optional[List[str]] = None) -> Dict[str, any]:
        """Create augmentation pipeline for training datasets based on size requirements."""
        if dataset_size >= target_size:
            self.logger.info("Dataset already meets target size, no augmentation needed")
            return {
                'augmentation_factor': 0,
                'strategies': [],
                'total_augmented_size': dataset_size
            }
        
        # Calculate required augmentation factor
        augmentation_factor = max(1, (target_size - dataset_size) // dataset_size)
        
        # Default augmentation strategies if none provided
        if augmentation_strategies is None:
            augmentation_strategies = [
                'rotation', 'scaling', 'horizontal_flip',
                'time_stretch', 'frame_dropout', 'noise'
            ]
        
        # Adjust augmentation parameters based on dataset size
        if dataset_size < 100:
            # More aggressive augmentation for very small datasets
            self.rotation_range = 20.0
            self.scale_range = (0.7, 1.3)
            self.noise_std = 0.03
            self.flip_probability = 0.7
        elif dataset_size < 500:
            # Moderate augmentation for small datasets
            self.rotation_range = 15.0
            self.scale_range = (0.8, 1.2)
            self.noise_std = 0.02
            self.flip_probability = 0.5
        else:
            # Conservative augmentation for larger datasets
            self.rotation_range = 10.0
            self.scale_range = (0.9, 1.1)
            self.noise_std = 0.01
            self.flip_probability = 0.3
        
        pipeline_config = {
            'augmentation_factor': augmentation_factor,
            'strategies': augmentation_strategies,
            'total_augmented_size': dataset_size * (1 + augmentation_factor),
            'parameters': {
                'rotation_range': self.rotation_range,
                'scale_range': self.scale_range,
                'noise_std': self.noise_std,
                'temporal_stretch_range': self.temporal_stretch_range,
                'flip_probability': self.flip_probability
            }
        }
        
        self.logger.info(f"Created augmentation pipeline: {pipeline_config}")
        return pipeline_config
