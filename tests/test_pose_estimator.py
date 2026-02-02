"""Tests for pose estimation functionality."""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from gait_analysis.pose_estimation import PoseEstimator
from gait_analysis.utils.data_structures import PoseKeypoint, PoseSequence, TrackedPose


class TestPoseEstimator:
    """Test cases for PoseEstimator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pose_estimator = PoseEstimator(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
    
    def test_pose_estimator_initialization(self):
        """Test PoseEstimator initialization."""
        assert self.pose_estimator.min_detection_confidence == 0.5
        assert self.pose_estimator.min_tracking_confidence == 0.5
        assert self.pose_estimator.model_complexity == 1
        assert self.pose_estimator.pose is not None
    
    def test_create_empty_keypoints(self):
        """Test creation of empty keypoints."""
        empty_keypoints = self.pose_estimator._create_empty_keypoints()
        
        assert len(empty_keypoints) == 33  # MediaPipe has 33 landmarks
        for keypoint in empty_keypoints:
            assert isinstance(keypoint, PoseKeypoint)
            assert keypoint.x == 0.0
            assert keypoint.y == 0.0
            assert keypoint.z == 0.0
            assert keypoint.confidence == 0.0
    
    def test_calculate_frame_confidence(self):
        """Test frame confidence calculation."""
        # Test with confident keypoints
        confident_keypoints = [
            PoseKeypoint(x=100, y=200, z=50, confidence=0.9),
            PoseKeypoint(x=150, y=250, z=60, confidence=0.8),
            PoseKeypoint(x=200, y=300, z=70, confidence=0.7)
        ]
        confidence = self.pose_estimator._calculate_frame_confidence(confident_keypoints)
        assert confidence == pytest.approx(0.8, rel=1e-2)
        
        # Test with empty keypoints
        empty_confidence = self.pose_estimator._calculate_frame_confidence([])
        assert empty_confidence == 0.0
    
    def test_extract_poses_empty_frames(self):
        """Test pose extraction with empty frame list."""
        with pytest.raises(ValueError, match="No frames provided"):
            self.pose_estimator.extract_poses([])
    
    def test_validate_pose_sequence(self):
        """Test pose sequence validation."""
        # Create a valid pose sequence (30+ frames with good confidence)
        valid_keypoints = []
        valid_timestamps = []
        valid_confidences = []
        
        for i in range(35):  # More than 30 frames
            frame_keypoints = [
                PoseKeypoint(x=100+i, y=200+i, z=50, confidence=0.8)
                for _ in range(33)
            ]
            valid_keypoints.append(frame_keypoints)
            valid_timestamps.append(i / 30.0)
            valid_confidences.append(0.8)
        
        valid_sequence = PoseSequence(
            keypoints=valid_keypoints,
            timestamps=valid_timestamps,
            confidence_scores=valid_confidences
        )
        
        assert self.pose_estimator.validate_pose_sequence(valid_sequence) is True
        
        # Test invalid sequence (too short)
        short_keypoints = valid_keypoints[:10]  # Only 10 frames
        short_timestamps = valid_timestamps[:10]
        short_confidences = valid_confidences[:10]
        
        short_sequence = PoseSequence(
            keypoints=short_keypoints,
            timestamps=short_timestamps,
            confidence_scores=short_confidences
        )
        
        assert self.pose_estimator.validate_pose_sequence(short_sequence) is False
        
        # Test sequence with low confidence
        low_conf_confidences = [0.1] * 35  # Low confidence scores
        low_conf_sequence = PoseSequence(
            keypoints=valid_keypoints,
            timestamps=valid_timestamps,
            confidence_scores=low_conf_confidences
        )
        
        assert self.pose_estimator.validate_pose_sequence(low_conf_sequence) is False
    
    def test_filter_low_confidence_keypoints(self):
        """Test filtering of low confidence keypoints."""
        # Create keypoints with mixed confidence levels
        mixed_keypoints = [
            [
                PoseKeypoint(x=100, y=200, z=50, confidence=0.9),  # High confidence
                PoseKeypoint(x=150, y=250, z=60, confidence=0.1),  # Low confidence
                PoseKeypoint(x=200, y=300, z=70, confidence=0.8),  # High confidence
            ]
        ]
        
        original_sequence = PoseSequence(
            keypoints=mixed_keypoints,
            timestamps=[0.0],
            confidence_scores=[0.6]
        )
        
        filtered_sequence = self.pose_estimator.filter_low_confidence_keypoints(
            original_sequence, min_confidence=0.3
        )
        
        # Check that low confidence keypoint was replaced
        assert filtered_sequence.keypoints[0][0].confidence == 0.9  # Unchanged
        assert filtered_sequence.keypoints[0][1].confidence == 0.0  # Filtered out
        assert filtered_sequence.keypoints[0][2].confidence == 0.8  # Unchanged
    
    def test_calculate_confidence_with_tracked_poses(self):
        """Test confidence calculation for tracked poses."""
        # Create mock tracked poses
        pose1 = Mock(spec=TrackedPose)
        pose1.tracking_confidence = 0.8
        
        pose2 = Mock(spec=TrackedPose)
        pose2.tracking_confidence = 0.6
        
        poses = [pose1, pose2]
        
        overall_confidence = self.pose_estimator.calculate_confidence(poses)
        assert overall_confidence == pytest.approx(0.7, rel=1e-2)
        
        # Test with empty list
        empty_confidence = self.pose_estimator.calculate_confidence([])
        assert empty_confidence == 0.0
    
    def test_temporal_smoothing(self):
        """Test temporal smoothing functionality."""
        # Create a sequence with some jitter
        keypoints_sequence = []
        for i in range(10):
            frame_keypoints = []
            for j in range(33):
                # Add some noise to simulate jitter
                noise = np.random.normal(0, 2)
                frame_keypoints.append(PoseKeypoint(
                    x=100 + noise, 
                    y=200 + noise, 
                    z=50, 
                    confidence=0.8
                ))
            keypoints_sequence.append(frame_keypoints)
        
        smoothed_sequence = self.pose_estimator._apply_temporal_smoothing(
            keypoints_sequence, window_size=5
        )
        
        # Check that we get the same number of frames
        assert len(smoothed_sequence) == len(keypoints_sequence)
        
        # Check that each frame has the correct number of keypoints
        for frame in smoothed_sequence:
            assert len(frame) == 33
            for keypoint in frame:
                assert isinstance(keypoint, PoseKeypoint)
    
    def test_track_landmarks_empty_sequence(self):
        """Test tracking with empty pose sequence."""
        empty_sequence = PoseSequence(keypoints=[], timestamps=[], confidence_scores=[])
        
        with pytest.raises(ValueError, match="Empty pose sequence"):
            self.pose_estimator.track_landmarks(empty_sequence)
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self.pose_estimator, 'pose'):
            self.pose_estimator.pose.close()


def create_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test frame for pose estimation."""
    # Create a simple test image with a stick figure
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw a simple stick figure (this won't actually be detected by MediaPipe,
    # but serves as a valid frame format for testing)
    center_x, center_y = width // 2, height // 2
    
    # Head
    cv2.circle(frame, (center_x, center_y - 100), 30, (255, 255, 255), -1)
    
    # Body
    cv2.line(frame, (center_x, center_y - 70), (center_x, center_y + 50), (255, 255, 255), 5)
    
    # Arms
    cv2.line(frame, (center_x, center_y - 30), (center_x - 50, center_y), (255, 255, 255), 3)
    cv2.line(frame, (center_x, center_y - 30), (center_x + 50, center_y), (255, 255, 255), 3)
    
    # Legs
    cv2.line(frame, (center_x, center_y + 50), (center_x - 30, center_y + 120), (255, 255, 255), 3)
    cv2.line(frame, (center_x, center_y + 50), (center_x + 30, center_y + 120), (255, 255, 255), 3)
    
    return frame


class TestPoseEstimatorIntegration:
    """Integration tests for PoseEstimator with actual MediaPipe processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pose_estimator = PoseEstimator()
    
    def test_extract_poses_with_test_frames(self):
        """Test pose extraction with actual test frames."""
        # Create test frames
        test_frames = [create_test_frame() for _ in range(5)]
        
        # Extract poses (this will likely not detect poses in our simple stick figure,
        # but tests the pipeline)
        pose_sequence = self.pose_estimator.extract_poses(test_frames)
        
        # Verify structure
        assert isinstance(pose_sequence, PoseSequence)
        assert len(pose_sequence.keypoints) == 5
        assert len(pose_sequence.timestamps) == 5
        assert len(pose_sequence.confidence_scores) == 5
        
        # Each frame should have 33 keypoints
        for frame_keypoints in pose_sequence.keypoints:
            assert len(frame_keypoints) == 33
            for keypoint in frame_keypoints:
                assert isinstance(keypoint, PoseKeypoint)
    
    def test_track_landmarks_integration(self):
        """Test landmark tracking with actual pose sequence."""
        # Create test frames and extract poses
        test_frames = [create_test_frame() for _ in range(10)]
        pose_sequence = self.pose_estimator.extract_poses(test_frames)
        
        # Track landmarks
        tracked_pose = self.pose_estimator.track_landmarks(pose_sequence)
        
        # Verify structure
        assert isinstance(tracked_pose, TrackedPose)
        assert isinstance(tracked_pose.pose_sequence, PoseSequence)
        assert tracked_pose.tracking_id == 1
        assert 0.0 <= tracked_pose.tracking_confidence <= 1.0
        
        # Verify smoothed sequence has same structure
        smoothed_sequence = tracked_pose.pose_sequence
        assert len(smoothed_sequence.keypoints) == len(pose_sequence.keypoints)
        assert len(smoothed_sequence.timestamps) == len(pose_sequence.timestamps)
        assert len(smoothed_sequence.confidence_scores) == len(pose_sequence.confidence_scores)
    
    def test_pose_quality_metrics(self):
        """Test pose quality metrics calculation."""
        # Create a test pose sequence
        test_frames = [create_test_frame() for _ in range(15)]
        pose_sequence = self.pose_estimator.extract_poses(test_frames)
        tracked_pose = self.pose_estimator.track_landmarks(pose_sequence)
        
        # Get quality metrics
        metrics = self.pose_estimator.get_pose_quality_metrics(tracked_pose)
        
        # Verify metrics structure
        expected_keys = ['overall_confidence', 'temporal_consistency', 'completeness', 'stability']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)
            assert 0.0 <= metrics[key] <= 1.0
    
    def test_temporal_consistency_maintenance(self):
        """Test temporal consistency maintenance functionality."""
        # Create keypoints with some outliers
        keypoints_sequence = []
        for i in range(10):
            frame_keypoints = []
            for j in range(33):
                # Add an outlier in the middle frame
                if i == 5 and j == 0:  # First keypoint in middle frame
                    frame_keypoints.append(PoseKeypoint(
                        x=1000,  # Large outlier
                        y=1000,
                        z=50,
                        confidence=0.8
                    ))
                else:
                    frame_keypoints.append(PoseKeypoint(
                        x=100 + i,  # Smooth progression
                        y=200 + i,
                        z=50,
                        confidence=0.8
                    ))
            keypoints_sequence.append(frame_keypoints)
        
        # Apply temporal consistency
        consistent_sequence = self.pose_estimator._maintain_temporal_consistency(keypoints_sequence)
        
        # Check that outlier was corrected
        corrected_keypoint = consistent_sequence[5][0]
        assert abs(corrected_keypoint.x - 1000) > 100  # Should be different from outlier
        assert corrected_keypoint.confidence < 0.8  # Confidence should be reduced
    
    def test_keypoint_interpolation(self):
        """Test keypoint interpolation functionality."""
        # Create sequence with missing keypoints
        keypoints_sequence = []
        for i in range(10):
            frame_keypoints = []
            for j in range(33):
                # Make middle frames have low confidence for first keypoint
                if 3 <= i <= 6 and j == 0:
                    frame_keypoints.append(PoseKeypoint(x=0, y=0, z=0, confidence=0.1))
                else:
                    frame_keypoints.append(PoseKeypoint(
                        x=100 + i * 10,
                        y=200 + i * 10,
                        z=50,
                        confidence=0.8
                    ))
            keypoints_sequence.append(frame_keypoints)
        
        # Apply interpolation
        interpolated_sequence = self.pose_estimator._interpolate_missing_keypoints(keypoints_sequence)
        
        # Check that missing keypoints were interpolated
        for i in range(3, 7):
            interpolated_kp = interpolated_sequence[i][0]
            assert interpolated_kp.confidence > 0.1  # Should have some confidence
            assert interpolated_kp.x > 0  # Should have interpolated position
            assert interpolated_kp.y > 0
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self.pose_estimator, 'pose'):
            self.pose_estimator.pose.close()