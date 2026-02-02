"""Tests for the FeatureExtractor class."""

import pytest
import numpy as np
from gait_analysis.feature_extraction import FeatureExtractor
from gait_analysis.utils.data_structures import (
    PoseKeypoint, PoseSequence, TrackedPose, GaitCycle
)


class TestFeatureExtractor:
    """Test cases for FeatureExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.feature_extractor = FeatureExtractor(fps=30.0)
        
    def create_sample_pose_sequence(self, n_frames=60):
        """Create a sample pose sequence for testing."""
        keypoints_sequence = []
        timestamps = []
        confidence_scores = []
        
        for frame_idx in range(n_frames):
            frame_keypoints = []
            
            # Create 33 keypoints (MediaPipe standard)
            for kp_idx in range(33):
                # Simulate walking motion with some variation
                x = 100 + np.sin(frame_idx * 0.1) * 10 + np.random.normal(0, 2)
                y = 200 + np.cos(frame_idx * 0.1) * 5 + np.random.normal(0, 1)
                z = 0 + np.random.normal(0, 1)
                confidence = 0.8 + np.random.normal(0, 0.1)
                confidence = max(0.0, min(1.0, confidence))
                
                frame_keypoints.append(PoseKeypoint(x=x, y=y, z=z, confidence=confidence))
            
            keypoints_sequence.append(frame_keypoints)
            timestamps.append(frame_idx / 30.0)
            confidence_scores.append(0.8)
        
        return PoseSequence(
            keypoints=keypoints_sequence,
            timestamps=timestamps,
            confidence_scores=confidence_scores
        )
    
    def test_feature_extractor_initialization(self):
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor(fps=25.0, min_cycle_duration=1.0, max_cycle_duration=3.0)
        
        assert extractor.fps == 25.0
        assert extractor.min_cycle_duration == 1.0
        assert extractor.max_cycle_duration == 3.0
        assert len(extractor.landmark_indices) > 0
    
    def test_extract_spatiotemporal_features(self):
        """Test spatiotemporal feature extraction."""
        pose_sequence = self.create_sample_pose_sequence(60)
        tracked_pose = TrackedPose(
            pose_sequence=pose_sequence,
            tracking_id=1,
            tracking_confidence=0.8
        )
        
        features = self.feature_extractor.extract_spatiotemporal_features(tracked_pose)
        
        # Check that features are extracted
        assert features.shape[0] == 60  # Number of frames
        assert features.shape[1] > 0    # Number of features per frame
        assert not np.isnan(features).all()  # Features should not be all NaN
    
    def test_extract_spatiotemporal_features_empty_sequence(self):
        """Test feature extraction with empty pose sequence."""
        empty_sequence = PoseSequence(keypoints=[], timestamps=[], confidence_scores=[])
        tracked_pose = TrackedPose(
            pose_sequence=empty_sequence,
            tracking_id=1,
            tracking_confidence=0.0
        )
        
        with pytest.raises(ValueError, match="Empty pose sequence provided"):
            self.feature_extractor.extract_spatiotemporal_features(tracked_pose)
    
    def test_segment_gait_cycles(self):
        """Test gait cycle segmentation."""
        pose_sequence = self.create_sample_pose_sequence(120)  # 4 seconds of data
        tracked_pose = TrackedPose(
            pose_sequence=pose_sequence,
            tracking_id=1,
            tracking_confidence=0.8
        )
        
        # Extract features first
        features = self.feature_extractor.extract_spatiotemporal_features(tracked_pose)
        
        # Segment gait cycles
        cycles = self.feature_extractor.segment_gait_cycles(features, pose_sequence)
        
        # Check that cycles are detected
        assert isinstance(cycles, list)
        
        # If cycles are detected, validate their properties
        for cycle in cycles:
            assert isinstance(cycle, GaitCycle)
            assert cycle.start_frame < cycle.end_frame
            assert cycle.cycle_time > 0
            assert cycle.step_count > 0
    
    def test_segment_gait_cycles_empty_features(self):
        """Test gait cycle segmentation with empty features."""
        empty_features = np.array([]).reshape(0, 0)
        empty_sequence = PoseSequence(keypoints=[], timestamps=[], confidence_scores=[])
        
        cycles = self.feature_extractor.segment_gait_cycles(empty_features, empty_sequence)
        
        assert cycles == []
    
    def test_normalize_features(self):
        """Test feature normalization."""
        # Create sample features
        features = np.random.randn(100, 50)  # 100 samples, 50 features
        
        normalized_features = self.feature_extractor.normalize_features(features)
        
        # Check shape is preserved
        assert normalized_features.shape == features.shape
        
        # Check that features are normalized (should have reasonable range)
        assert not np.isnan(normalized_features).any()
        assert not np.isinf(normalized_features).any()
    
    def test_normalize_features_empty(self):
        """Test feature normalization with empty array."""
        empty_features = np.array([]).reshape(0, 0)
        
        normalized = self.feature_extractor.normalize_features(empty_features)
        
        assert normalized.shape == empty_features.shape
    
    def test_normalize_features_with_nan(self):
        """Test feature normalization with NaN values."""
        features = np.random.randn(50, 20)
        features[10:15, 5:10] = np.nan  # Add some NaN values
        
        normalized_features = self.feature_extractor.normalize_features(features)
        
        # Check that NaN values are handled
        assert not np.isnan(normalized_features).any()
        assert normalized_features.shape == features.shape
    
    def test_landmark_indices_completeness(self):
        """Test that all required landmark indices are defined."""
        required_landmarks = [
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index', 'nose',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist'
        ]
        
        for landmark in required_landmarks:
            assert landmark in self.feature_extractor.landmark_indices
            assert isinstance(self.feature_extractor.landmark_indices[landmark], int)
            assert 0 <= self.feature_extractor.landmark_indices[landmark] < 33
    
    def test_feature_extraction_consistency(self):
        """Test that feature extraction produces consistent results."""
        pose_sequence = self.create_sample_pose_sequence(30)
        tracked_pose = TrackedPose(
            pose_sequence=pose_sequence,
            tracking_id=1,
            tracking_confidence=0.8
        )
        
        # Extract features twice
        features1 = self.feature_extractor.extract_spatiotemporal_features(tracked_pose)
        features2 = self.feature_extractor.extract_spatiotemporal_features(tracked_pose)
        
        # Results should be identical for the same input
        np.testing.assert_array_equal(features1, features2)
    
    def test_gait_cycle_validation(self):
        """Test that detected gait cycles have valid properties."""
        pose_sequence = self.create_sample_pose_sequence(150)  # 5 seconds of data
        tracked_pose = TrackedPose(
            pose_sequence=pose_sequence,
            tracking_id=1,
            tracking_confidence=0.8
        )
        
        features = self.feature_extractor.extract_spatiotemporal_features(tracked_pose)
        cycles = self.feature_extractor.segment_gait_cycles(features, pose_sequence)
        
        for cycle in cycles:
            # Validate cycle duration is within expected range
            assert self.feature_extractor.min_cycle_duration <= cycle.cycle_time <= self.feature_extractor.max_cycle_duration
            
            # Validate frame indices
            assert 0 <= cycle.start_frame < len(pose_sequence.keypoints)
            assert cycle.start_frame < cycle.end_frame <= len(pose_sequence.keypoints)
            
            # Validate step count
            assert cycle.step_count > 0