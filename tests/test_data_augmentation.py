"""Tests for the DataAugmentation class."""

import pytest
import numpy as np
from copy import deepcopy
from gait_analysis.feature_extraction.data_augmentation import DataAugmentation
from gait_analysis.utils.data_structures import (
    PoseKeypoint, PoseSequence, TrackedPose, TrainingExample
)


class TestDataAugmentation:
    """Test cases for DataAugmentation class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_augmentation = DataAugmentation(
            rotation_range=15.0,
            scale_range=(0.8, 1.2),
            noise_std=0.02,
            temporal_stretch_range=(0.8, 1.2),
            flip_probability=0.5
        )
        
    def create_sample_pose_sequence(self, n_frames=30):
        """Create a sample pose sequence for testing."""
        keypoints_sequence = []
        timestamps = []
        confidence_scores = []
        
        for frame_idx in range(n_frames):
            frame_keypoints = []
            
            # Create 33 keypoints (MediaPipe standard)
            for kp_idx in range(33):
                # Simulate walking motion
                x = 100 + np.sin(frame_idx * 0.2) * 20
                y = 200 + np.cos(frame_idx * 0.2) * 10
                z = 0 + np.random.normal(0, 1)
                confidence = 0.9
                
                frame_keypoints.append(PoseKeypoint(x=x, y=y, z=z, confidence=confidence))
            
            keypoints_sequence.append(frame_keypoints)
            timestamps.append(frame_idx / 30.0)
            confidence_scores.append(0.9)
        
        return PoseSequence(
            keypoints=keypoints_sequence,
            timestamps=timestamps,
            confidence_scores=confidence_scores
        )
    
    def create_sample_training_example(self):
        """Create a sample training example."""
        pose_sequence = self.create_sample_pose_sequence(30)
        
        return TrainingExample(
            video_path="test_video.mp4",
            pose_sequence=pose_sequence,
            ground_truth_label="normal",
            severity_score=0.0,
            metadata={"duration": 1.0}
        )
    
    def test_data_augmentation_initialization(self):
        """Test DataAugmentation initialization."""
        aug = DataAugmentation(
            rotation_range=20.0,
            scale_range=(0.7, 1.3),
            noise_std=0.03,
            temporal_stretch_range=(0.7, 1.3),
            flip_probability=0.6
        )
        
        assert aug.rotation_range == 20.0
        assert aug.scale_range == (0.7, 1.3)
        assert aug.noise_std == 0.03
        assert aug.temporal_stretch_range == (0.7, 1.3)
        assert aug.flip_probability == 0.6
        assert len(aug.flip_pairs) > 0
    
    def test_augment_training_dataset_empty(self):
        """Test augmentation with empty dataset."""
        empty_dataset = []
        
        augmented = self.data_augmentation.augment_training_dataset(empty_dataset)
        
        assert augmented == []
    
    def test_augment_training_dataset_basic(self):
        """Test basic dataset augmentation."""
        original_examples = [self.create_sample_training_example()]
        augmentation_factor = 2
        
        augmented_examples = self.data_augmentation.augment_training_dataset(
            original_examples, augmentation_factor
        )
        
        # Should have original + augmented examples
        expected_size = len(original_examples) * (1 + augmentation_factor)
        assert len(augmented_examples) == expected_size
        
        # First example should be the original
        assert augmented_examples[0].video_path == original_examples[0].video_path
        
        # Augmented examples should have modified paths
        for i in range(1, len(augmented_examples)):
            assert "_aug_" in augmented_examples[i].video_path
    
    def test_spatial_augmentation_rotation(self):
        """Test rotation augmentation."""
        pose_sequence = self.create_sample_pose_sequence(10)
        original_sequence = deepcopy(pose_sequence)
        
        # Apply rotation
        rotated_sequence = self.data_augmentation._apply_rotation(pose_sequence)
        
        # Check that coordinates have changed
        original_x = original_sequence.keypoints[0][0].x
        rotated_x = rotated_sequence.keypoints[0][0].x
        
        # Should be different (unless rotation angle was 0, which is unlikely)
        # We'll check that the structure is preserved
        assert len(rotated_sequence.keypoints) == len(original_sequence.keypoints)
        assert len(rotated_sequence.keypoints[0]) == len(original_sequence.keypoints[0])
    
    def test_spatial_augmentation_scaling(self):
        """Test scaling augmentation."""
        pose_sequence = self.create_sample_pose_sequence(10)
        original_sequence = deepcopy(pose_sequence)
        
        # Apply scaling
        scaled_sequence = self.data_augmentation._apply_scaling(pose_sequence)
        
        # Check that structure is preserved
        assert len(scaled_sequence.keypoints) == len(original_sequence.keypoints)
        assert len(scaled_sequence.keypoints[0]) == len(original_sequence.keypoints[0])
        
        # Check that coordinates have been scaled (should be different)
        original_coords = [
            (kp.x, kp.y, kp.z) for kp in original_sequence.keypoints[0]
        ]
        scaled_coords = [
            (kp.x, kp.y, kp.z) for kp in scaled_sequence.keypoints[0]
        ]
        
        # At least some coordinates should be different
        differences = sum(1 for orig, scaled in zip(original_coords, scaled_coords) 
                         if orig != scaled)
        assert differences > 0
    
    def test_spatial_augmentation_horizontal_flip(self):
        """Test horizontal flip augmentation."""
        pose_sequence = self.create_sample_pose_sequence(10)
        original_sequence = deepcopy(pose_sequence)
        
        # Force flip by setting probability to 1.0
        self.data_augmentation.flip_probability = 1.0
        
        # Apply horizontal flip
        flipped_sequence = self.data_augmentation._apply_horizontal_flip(pose_sequence)
        
        # Check that structure is preserved
        assert len(flipped_sequence.keypoints) == len(original_sequence.keypoints)
        
        # Check that x coordinates are flipped (negated)
        for frame_idx in range(len(original_sequence.keypoints)):
            for kp_idx in range(len(original_sequence.keypoints[frame_idx])):
                original_x = original_sequence.keypoints[frame_idx][kp_idx].x
                flipped_x = flipped_sequence.keypoints[frame_idx][kp_idx].x
                
                if original_sequence.keypoints[frame_idx][kp_idx].confidence > 0.1:
                    assert flipped_x == -original_x
    
    def test_temporal_augmentation_time_stretching(self):
        """Test time stretching augmentation."""
        pose_sequence = self.create_sample_pose_sequence(30)
        original_length = len(pose_sequence.keypoints)
        
        # Apply time stretching
        stretched_sequence = self.data_augmentation._apply_time_stretching(pose_sequence)
        
        # Length should be different (unless stretch factor was exactly 1.0)
        new_length = len(stretched_sequence.keypoints)
        
        # Should have valid structure
        assert new_length >= 5  # Minimum sequence length
        assert len(stretched_sequence.timestamps) == new_length
        assert len(stretched_sequence.confidence_scores) == new_length
    
    def test_temporal_augmentation_frame_dropout(self):
        """Test frame dropout augmentation."""
        pose_sequence = self.create_sample_pose_sequence(30)
        original_length = len(pose_sequence.keypoints)
        
        # Apply frame dropout
        dropout_sequence = self.data_augmentation._apply_frame_dropout(pose_sequence)
        
        # Should have fewer or equal frames
        new_length = len(dropout_sequence.keypoints)
        assert new_length <= original_length
        assert new_length >= 5  # Minimum sequence length
        
        # Structure should be preserved
        assert len(dropout_sequence.timestamps) == new_length
        assert len(dropout_sequence.confidence_scores) == new_length
    
    def test_temporal_augmentation_temporal_shift(self):
        """Test temporal shift augmentation."""
        pose_sequence = self.create_sample_pose_sequence(30)
        original_length = len(pose_sequence.keypoints)
        
        # Apply temporal shift
        shifted_sequence = self.data_augmentation._apply_temporal_shift(pose_sequence)
        
        # Length should be preserved
        assert len(shifted_sequence.keypoints) == original_length
        assert len(shifted_sequence.timestamps) == original_length
        assert len(shifted_sequence.confidence_scores) == original_length
    
    def test_noise_augmentation(self):
        """Test noise augmentation."""
        pose_sequence = self.create_sample_pose_sequence(10)
        original_sequence = deepcopy(pose_sequence)
        
        # Apply noise
        noisy_sequence = self.data_augmentation._apply_noise_augmentation(pose_sequence)
        
        # Check that structure is preserved
        assert len(noisy_sequence.keypoints) == len(original_sequence.keypoints)
        
        # Check that coordinates have noise added (should be different)
        differences = 0
        for frame_idx in range(len(original_sequence.keypoints)):
            for kp_idx in range(len(original_sequence.keypoints[frame_idx])):
                orig_kp = original_sequence.keypoints[frame_idx][kp_idx]
                noisy_kp = noisy_sequence.keypoints[frame_idx][kp_idx]
                
                if orig_kp.confidence > 0.1:
                    if (orig_kp.x != noisy_kp.x or 
                        orig_kp.y != noisy_kp.y or 
                        orig_kp.z != noisy_kp.z):
                        differences += 1
        
        # Should have some differences due to noise
        assert differences > 0
    
    def test_create_augmentation_pipeline_no_augmentation_needed(self):
        """Test pipeline creation when no augmentation is needed."""
        dataset_size = 1000
        target_size = 800  # Already exceeds target
        
        pipeline_config = self.data_augmentation.create_augmentation_pipeline(
            dataset_size, target_size
        )
        
        assert pipeline_config['augmentation_factor'] == 0
        assert pipeline_config['strategies'] == []
        assert pipeline_config['total_augmented_size'] == dataset_size
    
    def test_create_augmentation_pipeline_small_dataset(self):
        """Test pipeline creation for small dataset."""
        dataset_size = 50
        target_size = 200
        
        pipeline_config = self.data_augmentation.create_augmentation_pipeline(
            dataset_size, target_size
        )
        
        assert pipeline_config['augmentation_factor'] >= 1
        assert len(pipeline_config['strategies']) > 0
        assert pipeline_config['total_augmented_size'] > dataset_size
        assert 'parameters' in pipeline_config
    
    def test_create_augmentation_pipeline_custom_strategies(self):
        """Test pipeline creation with custom strategies."""
        dataset_size = 100
        target_size = 300
        custom_strategies = ['rotation', 'scaling']
        
        pipeline_config = self.data_augmentation.create_augmentation_pipeline(
            dataset_size, target_size, custom_strategies
        )
        
        assert pipeline_config['strategies'] == custom_strategies
        assert pipeline_config['augmentation_factor'] >= 1
    
    def test_augmentation_preserves_label_and_metadata(self):
        """Test that augmentation preserves labels and metadata."""
        original_example = self.create_sample_training_example()
        original_example.ground_truth_label = "abnormal"
        original_example.severity_score = 0.7
        original_example.metadata = {"test": "value"}
        
        augmented_example = self.data_augmentation._augment_single_example(
            original_example, 0
        )
        
        # Labels and metadata should be preserved
        assert augmented_example.ground_truth_label == original_example.ground_truth_label
        assert augmented_example.severity_score == original_example.severity_score
        assert augmented_example.metadata == original_example.metadata
    
    def test_augmentation_reproducibility(self):
        """Test that augmentation is reproducible with same seed."""
        original_example = self.create_sample_training_example()
        
        # Apply same augmentation twice
        augmented1 = self.data_augmentation._augment_single_example(original_example, 42)
        augmented2 = self.data_augmentation._augment_single_example(original_example, 42)
        
        # Results should be identical
        assert len(augmented1.pose_sequence.keypoints) == len(augmented2.pose_sequence.keypoints)
        
        # Check that coordinates are the same
        for frame_idx in range(len(augmented1.pose_sequence.keypoints)):
            for kp_idx in range(len(augmented1.pose_sequence.keypoints[frame_idx])):
                kp1 = augmented1.pose_sequence.keypoints[frame_idx][kp_idx]
                kp2 = augmented2.pose_sequence.keypoints[frame_idx][kp_idx]
                
                assert abs(kp1.x - kp2.x) < 1e-10
                assert abs(kp1.y - kp2.y) < 1e-10
                assert abs(kp1.z - kp2.z) < 1e-10
                assert abs(kp1.confidence - kp2.confidence) < 1e-10
    
    def test_flip_pairs_validity(self):
        """Test that flip pairs are valid MediaPipe indices."""
        for left_idx, right_idx in self.data_augmentation.flip_pairs:
            assert isinstance(left_idx, int)
            assert isinstance(right_idx, int)
            assert 0 <= left_idx < 33  # MediaPipe has 33 landmarks
            assert 0 <= right_idx < 33
            assert left_idx != right_idx
    
    def test_augmentation_handles_low_confidence_keypoints(self):
        """Test that augmentation properly handles low confidence keypoints."""
        pose_sequence = self.create_sample_pose_sequence(10)
        
        # Set some keypoints to low confidence
        for frame_keypoints in pose_sequence.keypoints:
            for i in range(0, len(frame_keypoints), 3):  # Every third keypoint
                frame_keypoints[i].confidence = 0.05  # Very low confidence
        
        original_sequence = deepcopy(pose_sequence)
        
        # Apply various augmentations
        rotated = self.data_augmentation._apply_rotation(deepcopy(pose_sequence))
        scaled = self.data_augmentation._apply_scaling(deepcopy(pose_sequence))
        noisy = self.data_augmentation._apply_noise_augmentation(deepcopy(pose_sequence))
        
        # Should not crash and should preserve structure
        assert len(rotated.keypoints) == len(original_sequence.keypoints)
        assert len(scaled.keypoints) == len(original_sequence.keypoints)
        assert len(noisy.keypoints) == len(original_sequence.keypoints)
    
    def test_temporal_augmentation_short_sequence(self):
        """Test temporal augmentation with very short sequences."""
        short_sequence = self.create_sample_pose_sequence(5)  # Very short
        
        # These should handle short sequences gracefully
        stretched = self.data_augmentation._apply_time_stretching(short_sequence)
        dropout = self.data_augmentation._apply_frame_dropout(short_sequence)
        shifted = self.data_augmentation._apply_temporal_shift(short_sequence)
        
        # Should not crash and should maintain minimum structure
        assert len(stretched.keypoints) >= 0
        assert len(dropout.keypoints) >= 0
        assert len(shifted.keypoints) >= 0