"""MediaPipe-based pose estimation for gait analysis."""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple, Dict
import logging

from ..utils.data_structures import PoseKeypoint, PoseSequence, TrackedPose


class PoseEstimator:
    """MediaPipe-based pose estimator for extracting human pose keypoints."""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1):
        """
        Initialize the pose estimator.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            model_complexity: Model complexity (0=lite, 1=full, 2=heavy)
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        
        # Initialize MediaPipe pose solution
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # MediaPipe drawing utilities (for visualization if needed)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def extract_poses(self, frames: List[np.ndarray]) -> PoseSequence:
        """
        Extract pose sequences from video frames.
        
        Args:
            frames: List of video frames as numpy arrays
            
        Returns:
            PoseSequence containing keypoints for all frames
        """
        if not frames:
            raise ValueError("No frames provided for pose extraction")
            
        keypoints_sequence = []
        timestamps = []
        confidence_scores = []
        
        for frame_idx, frame in enumerate(frames):
            try:
                # Convert BGR to RGB (MediaPipe expects RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame
                results = self.pose.process(rgb_frame)
                
                # Extract keypoints if pose is detected
                if results.pose_landmarks:
                    frame_keypoints = self._extract_keypoints_from_landmarks(
                        results.pose_landmarks, 
                        frame.shape
                    )
                    frame_confidence = self._calculate_frame_confidence(frame_keypoints)
                else:
                    # No pose detected - create empty keypoints with zero confidence
                    frame_keypoints = self._create_empty_keypoints()
                    frame_confidence = 0.0
                
                keypoints_sequence.append(frame_keypoints)
                timestamps.append(frame_idx / 30.0)  # Assuming 30 FPS
                confidence_scores.append(frame_confidence)
                
            except Exception as e:
                self.logger.warning(f"Error processing frame {frame_idx}: {e}")
                # Add empty keypoints for failed frames
                keypoints_sequence.append(self._create_empty_keypoints())
                timestamps.append(frame_idx / 30.0)
                confidence_scores.append(0.0)
        
        return PoseSequence(
            keypoints=keypoints_sequence,
            timestamps=timestamps,
            confidence_scores=confidence_scores
        )
    
    def _extract_keypoints_from_landmarks(self, 
                                        landmarks, 
                                        frame_shape: Tuple[int, int, int]) -> List[PoseKeypoint]:
        """
        Extract keypoints from MediaPipe landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            List of PoseKeypoint objects
        """
        keypoints = []
        height, width = frame_shape[:2]
        
        for landmark in landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates
            x = landmark.x * width
            y = landmark.y * height
            z = landmark.z * width  # Z is relative to hip width
            
            # MediaPipe provides visibility as confidence
            confidence = landmark.visibility
            
            keypoints.append(PoseKeypoint(x=x, y=y, z=z, confidence=confidence))
        
        return keypoints
    
    def _create_empty_keypoints(self) -> List[PoseKeypoint]:
        """Create empty keypoints for frames where no pose is detected."""
        return [PoseKeypoint(x=0.0, y=0.0, z=0.0, confidence=0.0) for _ in range(33)]
    
    def _calculate_frame_confidence(self, keypoints: List[PoseKeypoint]) -> float:
        """
        Calculate overall confidence for a frame based on keypoint confidences.
        
        Args:
            keypoints: List of keypoints for the frame
            
        Returns:
            Average confidence score for the frame
        """
        if not keypoints:
            return 0.0
        
        confidences = [kp.confidence for kp in keypoints]
        return np.mean(confidences)
    
    def track_landmarks(self, pose_sequence: PoseSequence) -> TrackedPose:
        """
        Apply temporal consistency and tracking to pose sequence.
        
        Args:
            pose_sequence: Raw pose sequence from extraction
            
        Returns:
            TrackedPose with improved temporal consistency
        """
        if not pose_sequence.keypoints:
            raise ValueError("Empty pose sequence provided for tracking")
        
        # Step 1: Apply temporal smoothing to reduce jitter
        smoothed_keypoints = self._apply_temporal_smoothing(pose_sequence.keypoints)
        
        # Step 2: Apply advanced temporal consistency maintenance
        consistent_keypoints = self._maintain_temporal_consistency(smoothed_keypoints)
        
        # Step 3: Interpolate missing keypoints
        interpolated_keypoints = self._interpolate_missing_keypoints(consistent_keypoints)
        
        # Step 4: Calculate tracking confidence based on temporal consistency
        tracking_confidence = self._calculate_tracking_confidence(
            pose_sequence.keypoints, 
            interpolated_keypoints
        )
        
        # Step 5: Update confidence scores based on tracking quality
        updated_confidence_scores = self._update_confidence_scores(
            pose_sequence.confidence_scores,
            interpolated_keypoints
        )
        
        # Create enhanced pose sequence
        enhanced_sequence = PoseSequence(
            keypoints=interpolated_keypoints,
            timestamps=pose_sequence.timestamps,
            confidence_scores=updated_confidence_scores
        )
        
        return TrackedPose(
            pose_sequence=enhanced_sequence,
            tracking_id=1,  # Single person tracking for now
            tracking_confidence=tracking_confidence
        )
    
    def _apply_temporal_smoothing(self, 
                                keypoints_sequence: List[List[PoseKeypoint]],
                                window_size: int = 5) -> List[List[PoseKeypoint]]:
        """
        Apply temporal smoothing to reduce keypoint jitter.
        
        Args:
            keypoints_sequence: Raw keypoint sequence
            window_size: Size of smoothing window
            
        Returns:
            Smoothed keypoint sequence
        """
        if len(keypoints_sequence) < window_size:
            return keypoints_sequence
        
        smoothed_sequence = []
        
        for frame_idx in range(len(keypoints_sequence)):
            # Define smoothing window
            start_idx = max(0, frame_idx - window_size // 2)
            end_idx = min(len(keypoints_sequence), frame_idx + window_size // 2 + 1)
            
            # Get keypoints in window
            window_keypoints = keypoints_sequence[start_idx:end_idx]
            
            # Smooth each keypoint
            smoothed_frame_keypoints = []
            for kp_idx in range(33):  # 33 MediaPipe landmarks
                # Extract coordinates for this keypoint across the window
                x_coords = [frame[kp_idx].x for frame in window_keypoints 
                           if frame[kp_idx].confidence > 0.1]
                y_coords = [frame[kp_idx].y for frame in window_keypoints 
                           if frame[kp_idx].confidence > 0.1]
                z_coords = [frame[kp_idx].z for frame in window_keypoints 
                           if frame[kp_idx].confidence > 0.1]
                confidences = [frame[kp_idx].confidence for frame in window_keypoints]
                
                # Calculate smoothed coordinates
                if x_coords:  # If we have valid keypoints
                    smoothed_x = np.median(x_coords)
                    smoothed_y = np.median(y_coords)
                    smoothed_z = np.median(z_coords)
                    smoothed_confidence = np.mean(confidences)
                else:
                    # Use original keypoint if no valid ones in window
                    original_kp = keypoints_sequence[frame_idx][kp_idx]
                    smoothed_x = original_kp.x
                    smoothed_y = original_kp.y
                    smoothed_z = original_kp.z
                    smoothed_confidence = original_kp.confidence
                
                smoothed_frame_keypoints.append(PoseKeypoint(
                    x=smoothed_x,
                    y=smoothed_y,
                    z=smoothed_z,
                    confidence=smoothed_confidence
                ))
            
            smoothed_sequence.append(smoothed_frame_keypoints)
        
        return smoothed_sequence
    
    def _calculate_tracking_confidence(self, 
                                     original_keypoints: List[List[PoseKeypoint]],
                                     smoothed_keypoints: List[List[PoseKeypoint]]) -> float:
        """
        Calculate tracking confidence based on temporal consistency.
        
        Args:
            original_keypoints: Original keypoint sequence
            smoothed_keypoints: Smoothed keypoint sequence
            
        Returns:
            Tracking confidence score (0-1)
        """
        if not original_keypoints or not smoothed_keypoints:
            return 0.0
        
        # Calculate average confidence across all frames and keypoints
        total_confidence = 0.0
        valid_keypoints = 0
        
        for frame_idx in range(len(original_keypoints)):
            for kp_idx in range(len(original_keypoints[frame_idx])):
                confidence = original_keypoints[frame_idx][kp_idx].confidence
                if confidence > 0.1:  # Only count confident keypoints
                    total_confidence += confidence
                    valid_keypoints += 1
        
        if valid_keypoints == 0:
            return 0.0
        
        return total_confidence / valid_keypoints
    
    def calculate_confidence(self, poses: List[TrackedPose]) -> float:
        """
        Calculate overall confidence for a list of tracked poses.
        
        Args:
            poses: List of tracked poses
            
        Returns:
            Overall confidence score
        """
        if not poses:
            return 0.0
        
        confidences = [pose.tracking_confidence for pose in poses]
        return np.mean(confidences)
    
    def validate_pose_sequence(self, pose_sequence: PoseSequence) -> bool:
        """
        Validate pose sequence quality and completeness.
        
        Args:
            pose_sequence: Pose sequence to validate
            
        Returns:
            True if sequence is valid for analysis
        """
        if not pose_sequence.keypoints:
            return False
        
        # Check minimum sequence length (at least 1 second at 30fps)
        if len(pose_sequence.keypoints) < 30:
            return False
        
        # Check that we have sufficient confident keypoints
        confident_frames = 0
        for frame_confidence in pose_sequence.confidence_scores:
            if frame_confidence > 0.3:  # Minimum confidence threshold
                confident_frames += 1
        
        # Require at least 70% of frames to have confident pose detection
        confidence_ratio = confident_frames / len(pose_sequence.keypoints)
        return confidence_ratio >= 0.7
    
    def filter_low_confidence_keypoints(self, 
                                      pose_sequence: PoseSequence,
                                      min_confidence: float = 0.3) -> PoseSequence:
        """
        Filter out keypoints with low confidence scores.
        
        Args:
            pose_sequence: Input pose sequence
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered pose sequence
        """
        filtered_keypoints = []
        
        for frame_keypoints in pose_sequence.keypoints:
            filtered_frame = []
            for keypoint in frame_keypoints:
                if keypoint.confidence >= min_confidence:
                    filtered_frame.append(keypoint)
                else:
                    # Replace low-confidence keypoints with zero-confidence placeholders
                    filtered_frame.append(PoseKeypoint(x=0.0, y=0.0, z=0.0, confidence=0.0))
            filtered_keypoints.append(filtered_frame)
        
        return PoseSequence(
            keypoints=filtered_keypoints,
            timestamps=pose_sequence.timestamps,
            confidence_scores=pose_sequence.confidence_scores
        )
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()
    
    def _maintain_temporal_consistency(self, 
                                     keypoints_sequence: List[List[PoseKeypoint]]) -> List[List[PoseKeypoint]]:
        """
        Maintain temporal consistency by detecting and correcting outliers.
        
        Args:
            keypoints_sequence: Smoothed keypoint sequence
            
        Returns:
            Temporally consistent keypoint sequence
        """
        if len(keypoints_sequence) < 3:
            return keypoints_sequence
        
        consistent_sequence = []
        
        for frame_idx in range(len(keypoints_sequence)):
            frame_keypoints = []
            
            for kp_idx in range(33):  # 33 MediaPipe landmarks
                current_kp = keypoints_sequence[frame_idx][kp_idx]
                
                # Skip if current keypoint has very low confidence
                if current_kp.confidence < 0.1:
                    frame_keypoints.append(current_kp)
                    continue
                
                # Get neighboring frames for consistency check
                prev_kp = None
                next_kp = None
                
                if frame_idx > 0:
                    prev_kp = keypoints_sequence[frame_idx - 1][kp_idx]
                if frame_idx < len(keypoints_sequence) - 1:
                    next_kp = keypoints_sequence[frame_idx + 1][kp_idx]
                
                # Check for temporal consistency
                consistent_kp = self._check_keypoint_consistency(current_kp, prev_kp, next_kp)
                frame_keypoints.append(consistent_kp)
            
            consistent_sequence.append(frame_keypoints)
        
        return consistent_sequence
    
    def _check_keypoint_consistency(self, 
                                  current: PoseKeypoint,
                                  previous: Optional[PoseKeypoint],
                                  next_kp: Optional[PoseKeypoint],
                                  max_displacement: float = 50.0) -> PoseKeypoint:
        """
        Check if a keypoint is consistent with its temporal neighbors.
        
        Args:
            current: Current frame keypoint
            previous: Previous frame keypoint
            next_kp: Next frame keypoint
            max_displacement: Maximum allowed displacement between frames
            
        Returns:
            Corrected keypoint if inconsistent, original otherwise
        """
        # If we don't have neighbors, return current keypoint
        if not previous and not next_kp:
            return current
        
        # Calculate expected position based on neighbors
        expected_x = current.x
        expected_y = current.y
        expected_z = current.z
        
        if previous and next_kp:
            # Interpolate between previous and next
            if previous.confidence > 0.1 and next_kp.confidence > 0.1:
                expected_x = (previous.x + next_kp.x) / 2
                expected_y = (previous.y + next_kp.y) / 2
                expected_z = (previous.z + next_kp.z) / 2
        elif previous and previous.confidence > 0.1:
            # Use previous position as reference
            expected_x = previous.x
            expected_y = previous.y
            expected_z = previous.z
        elif next_kp and next_kp.confidence > 0.1:
            # Use next position as reference
            expected_x = next_kp.x
            expected_y = next_kp.y
            expected_z = next_kp.z
        
        # Calculate displacement from expected position
        displacement = np.sqrt(
            (current.x - expected_x) ** 2 + 
            (current.y - expected_y) ** 2
        )
        
        # If displacement is too large, use expected position with reduced confidence
        if displacement > max_displacement and current.confidence > 0.1:
            return PoseKeypoint(
                x=expected_x,
                y=expected_y,
                z=expected_z,
                confidence=current.confidence * 0.5  # Reduce confidence for corrected points
            )
        
        return current
    
    def _interpolate_missing_keypoints(self, 
                                     keypoints_sequence: List[List[PoseKeypoint]]) -> List[List[PoseKeypoint]]:
        """
        Interpolate missing or low-confidence keypoints using temporal information.
        
        Args:
            keypoints_sequence: Keypoint sequence with potential gaps
            
        Returns:
            Sequence with interpolated keypoints
        """
        if len(keypoints_sequence) < 2:
            return keypoints_sequence
        
        interpolated_sequence = []
        
        for frame_idx in range(len(keypoints_sequence)):
            frame_keypoints = []
            
            for kp_idx in range(33):
                current_kp = keypoints_sequence[frame_idx][kp_idx]
                
                # If keypoint has good confidence, keep it
                if current_kp.confidence >= 0.3:
                    frame_keypoints.append(current_kp)
                    continue
                
                # Try to interpolate from neighboring frames
                interpolated_kp = self._interpolate_keypoint(
                    keypoints_sequence, frame_idx, kp_idx
                )
                frame_keypoints.append(interpolated_kp)
            
            interpolated_sequence.append(frame_keypoints)
        
        return interpolated_sequence
    
    def _interpolate_keypoint(self, 
                            keypoints_sequence: List[List[PoseKeypoint]],
                            frame_idx: int,
                            kp_idx: int,
                            search_radius: int = 5) -> PoseKeypoint:
        """
        Interpolate a single keypoint using temporal neighbors.
        
        Args:
            keypoints_sequence: Full keypoint sequence
            frame_idx: Current frame index
            kp_idx: Keypoint index to interpolate
            search_radius: Number of frames to search in each direction
            
        Returns:
            Interpolated keypoint
        """
        current_kp = keypoints_sequence[frame_idx][kp_idx]
        
        # Find valid keypoints before and after current frame
        before_kp = None
        after_kp = None
        before_idx = -1
        after_idx = -1
        
        # Search backwards
        for i in range(max(0, frame_idx - search_radius), frame_idx):
            kp = keypoints_sequence[i][kp_idx]
            if kp.confidence >= 0.3:
                before_kp = kp
                before_idx = i
                break
        
        # Search forwards
        for i in range(frame_idx + 1, min(len(keypoints_sequence), frame_idx + search_radius + 1)):
            kp = keypoints_sequence[i][kp_idx]
            if kp.confidence >= 0.3:
                after_kp = kp
                after_idx = i
                break
        
        # Interpolate based on available neighbors
        if before_kp and after_kp:
            # Linear interpolation between before and after
            total_frames = after_idx - before_idx
            current_offset = frame_idx - before_idx
            weight = current_offset / total_frames
            
            interpolated_x = before_kp.x + weight * (after_kp.x - before_kp.x)
            interpolated_y = before_kp.y + weight * (after_kp.y - before_kp.y)
            interpolated_z = before_kp.z + weight * (after_kp.z - before_kp.z)
            interpolated_confidence = min(before_kp.confidence, after_kp.confidence) * 0.7
            
            return PoseKeypoint(
                x=interpolated_x,
                y=interpolated_y,
                z=interpolated_z,
                confidence=interpolated_confidence
            )
        
        elif before_kp:
            # Use previous keypoint with reduced confidence
            return PoseKeypoint(
                x=before_kp.x,
                y=before_kp.y,
                z=before_kp.z,
                confidence=before_kp.confidence * 0.5
            )
        
        elif after_kp:
            # Use next keypoint with reduced confidence
            return PoseKeypoint(
                x=after_kp.x,
                y=after_kp.y,
                z=after_kp.z,
                confidence=after_kp.confidence * 0.5
            )
        
        else:
            # No valid neighbors found, return original with very low confidence
            return PoseKeypoint(
                x=current_kp.x,
                y=current_kp.y,
                z=current_kp.z,
                confidence=0.1
            )
    
    def _update_confidence_scores(self, 
                                original_scores: List[float],
                                processed_keypoints: List[List[PoseKeypoint]]) -> List[float]:
        """
        Update frame confidence scores based on processed keypoints.
        
        Args:
            original_scores: Original confidence scores
            processed_keypoints: Processed keypoint sequence
            
        Returns:
            Updated confidence scores
        """
        updated_scores = []
        
        for frame_idx, frame_keypoints in enumerate(processed_keypoints):
            # Calculate new frame confidence based on processed keypoints
            frame_confidence = self._calculate_frame_confidence(frame_keypoints)
            
            # Blend with original confidence (weighted average)
            if frame_idx < len(original_scores):
                original_confidence = original_scores[frame_idx]
                blended_confidence = 0.7 * frame_confidence + 0.3 * original_confidence
            else:
                blended_confidence = frame_confidence
            
            updated_scores.append(blended_confidence)
        
        return updated_scores
    
    def get_pose_quality_metrics(self, tracked_pose: TrackedPose) -> Dict[str, float]:
        """
        Calculate quality metrics for a tracked pose sequence.
        
        Args:
            tracked_pose: Tracked pose to analyze
            
        Returns:
            Dictionary of quality metrics
        """
        pose_sequence = tracked_pose.pose_sequence
        
        if not pose_sequence.keypoints:
            return {
                'overall_confidence': 0.0,
                'temporal_consistency': 0.0,
                'completeness': 0.0,
                'stability': 0.0
            }
        
        # Overall confidence
        overall_confidence = np.mean(pose_sequence.confidence_scores)
        
        # Temporal consistency (measure of smoothness)
        temporal_consistency = self._calculate_temporal_consistency(pose_sequence.keypoints)
        
        # Completeness (percentage of confident keypoints)
        completeness = self._calculate_completeness(pose_sequence.keypoints)
        
        # Stability (inverse of variance in keypoint positions)
        stability = self._calculate_stability(pose_sequence.keypoints)
        
        return {
            'overall_confidence': float(overall_confidence),
            'temporal_consistency': float(temporal_consistency),
            'completeness': float(completeness),
            'stability': float(stability)
        }
    
    def _calculate_temporal_consistency(self, keypoints_sequence: List[List[PoseKeypoint]]) -> float:
        """Calculate temporal consistency metric."""
        if len(keypoints_sequence) < 2:
            return 1.0
        
        total_consistency = 0.0
        valid_comparisons = 0
        
        for frame_idx in range(1, len(keypoints_sequence)):
            for kp_idx in range(33):
                current_kp = keypoints_sequence[frame_idx][kp_idx]
                prev_kp = keypoints_sequence[frame_idx - 1][kp_idx]
                
                if current_kp.confidence > 0.1 and prev_kp.confidence > 0.1:
                    # Calculate displacement between consecutive frames
                    displacement = np.sqrt(
                        (current_kp.x - prev_kp.x) ** 2 + 
                        (current_kp.y - prev_kp.y) ** 2
                    )
                    
                    # Convert to consistency score (lower displacement = higher consistency)
                    consistency = 1.0 / (1.0 + displacement / 10.0)  # Normalize by expected movement
                    total_consistency += consistency
                    valid_comparisons += 1
        
        return total_consistency / valid_comparisons if valid_comparisons > 0 else 0.0
    
    def _calculate_completeness(self, keypoints_sequence: List[List[PoseKeypoint]]) -> float:
        """Calculate completeness metric (percentage of confident keypoints)."""
        if not keypoints_sequence:
            return 0.0
        
        total_keypoints = len(keypoints_sequence) * 33
        confident_keypoints = 0
        
        for frame_keypoints in keypoints_sequence:
            for keypoint in frame_keypoints:
                if keypoint.confidence > 0.3:
                    confident_keypoints += 1
        
        return confident_keypoints / total_keypoints
    
    def _calculate_stability(self, keypoints_sequence: List[List[PoseKeypoint]]) -> float:
        """Calculate stability metric (inverse of position variance)."""
        if len(keypoints_sequence) < 3:
            return 1.0
        
        total_stability = 0.0
        valid_keypoints = 0
        
        for kp_idx in range(33):
            # Collect positions for this keypoint across all frames
            x_positions = []
            y_positions = []
            
            for frame_keypoints in keypoints_sequence:
                keypoint = frame_keypoints[kp_idx]
                if keypoint.confidence > 0.3:
                    x_positions.append(keypoint.x)
                    y_positions.append(keypoint.y)
            
            if len(x_positions) > 2:
                # Calculate variance and convert to stability score
                x_var = np.var(x_positions)
                y_var = np.var(y_positions)
                total_var = x_var + y_var
                
                # Convert to stability (lower variance = higher stability)
                stability = 1.0 / (1.0 + total_var / 1000.0)  # Normalize
                total_stability += stability
                valid_keypoints += 1
        
        return total_stability / valid_keypoints if valid_keypoints > 0 else 0.0