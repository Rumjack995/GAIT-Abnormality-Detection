"""Feature extraction pipeline for gait analysis."""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from scipy import signal
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ..utils.data_structures import (
    PoseSequence, TrackedPose, GaitCycle, PoseKeypoint
)


class FeatureExtractor:
    """
    Extracts spatiotemporal features from pose sequences for gait analysis.
    
    This class implements feature computation from pose sequences, gait cycle
    segmentation algorithms, and feature normalization methods as specified
    in Requirements 6.1 and 6.3.
    """
    
    def __init__(self, 
                 fps: float = 30.0,
                 min_cycle_duration: float = 0.8,
                 max_cycle_duration: float = 2.5):
        """
        Initialize the feature extractor.
        
        Args:
            fps: Frames per second of input video
            min_cycle_duration: Minimum gait cycle duration in seconds
            max_cycle_duration: Maximum gait cycle duration in seconds
        """
        self.fps = fps
        self.min_cycle_duration = min_cycle_duration
        self.max_cycle_duration = max_cycle_duration
        
        # MediaPipe landmark indices for key body parts
        self.landmark_indices = {
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot_index': 31,
            'right_foot_index': 32,
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Scalers for normalization (will be fitted during processing)
        self.spatial_scaler = StandardScaler()
        self.temporal_scaler = MinMaxScaler()
        
    def extract_spatiotemporal_features(self, poses: TrackedPose) -> np.ndarray:
        """
        Extract spatiotemporal features from pose sequences.
        
        Implements spatiotemporal feature computation as required by Requirement 6.1.
        
        Args:
            poses: TrackedPose containing pose sequence data
            
        Returns:
            Feature array of shape (n_frames, n_features)
        """
        pose_sequence = poses.pose_sequence
        
        if not pose_sequence.keypoints:
            raise ValueError("Empty pose sequence provided")
        
        # Extract different types of features
        spatial_features = self._extract_spatial_features(pose_sequence)
        temporal_features = self._extract_temporal_features(pose_sequence)
        kinematic_features = self._extract_kinematic_features(pose_sequence)
        
        # Combine all features
        combined_features = np.concatenate([
            spatial_features,
            temporal_features,
            kinematic_features
        ], axis=1)
        
        self.logger.info(f"Extracted features shape: {combined_features.shape}")
        return combined_features
    
    def _extract_spatial_features(self, pose_sequence: PoseSequence) -> np.ndarray:
        """
        Extract spatial features from pose keypoints.
        
        Args:
            pose_sequence: Input pose sequence
            
        Returns:
            Spatial features array
        """
        n_frames = len(pose_sequence.keypoints)
        spatial_features = []
        
        for frame_idx in range(n_frames):
            frame_keypoints = pose_sequence.keypoints[frame_idx]
            frame_features = []
            
            # 1. Joint positions (normalized by torso length)
            joint_positions = self._get_normalized_joint_positions(frame_keypoints)
            frame_features.extend(joint_positions)
            
            # 2. Joint angles
            joint_angles = self._calculate_joint_angles(frame_keypoints)
            frame_features.extend(joint_angles)
            
            # 3. Limb lengths
            limb_lengths = self._calculate_limb_lengths(frame_keypoints)
            frame_features.extend(limb_lengths)
            
            # 4. Body symmetry measures
            symmetry_measures = self._calculate_body_symmetry(frame_keypoints)
            frame_features.extend(symmetry_measures)
            
            spatial_features.append(frame_features)
        
        return np.array(spatial_features)
    
    def _extract_temporal_features(self, pose_sequence: PoseSequence) -> np.ndarray:
        """
        Extract temporal features from pose sequence.
        
        Args:
            pose_sequence: Input pose sequence
            
        Returns:
            Temporal features array
        """
        n_frames = len(pose_sequence.keypoints)
        temporal_features = []
        
        # Calculate velocities and accelerations for key joints
        key_joints = ['left_ankle', 'right_ankle', 'left_knee', 'right_knee']
        
        for frame_idx in range(n_frames):
            frame_features = []
            
            # Joint velocities
            velocities = self._calculate_joint_velocities(
                pose_sequence, frame_idx, key_joints
            )
            frame_features.extend(velocities)
            
            # Joint accelerations
            accelerations = self._calculate_joint_accelerations(
                pose_sequence, frame_idx, key_joints
            )
            frame_features.extend(accelerations)
            
            # Temporal consistency measures
            consistency = self._calculate_temporal_consistency_features(
                pose_sequence, frame_idx
            )
            frame_features.extend(consistency)
            
            temporal_features.append(frame_features)
        
        return np.array(temporal_features)
    
    def _extract_kinematic_features(self, pose_sequence: PoseSequence) -> np.ndarray:
        """
        Extract kinematic features related to gait patterns.
        
        Args:
            pose_sequence: Input pose sequence
            
        Returns:
            Kinematic features array
        """
        n_frames = len(pose_sequence.keypoints)
        kinematic_features = []
        
        for frame_idx in range(n_frames):
            frame_keypoints = pose_sequence.keypoints[frame_idx]
            frame_features = []
            
            # 1. Step width (distance between feet)
            step_width = self._calculate_step_width(frame_keypoints)
            frame_features.append(step_width)
            
            # 2. Stride length indicators
            stride_indicators = self._calculate_stride_indicators(
                pose_sequence, frame_idx
            )
            frame_features.extend(stride_indicators)
            
            # 3. Center of mass estimation
            com_features = self._calculate_center_of_mass_features(frame_keypoints)
            frame_features.extend(com_features)
            
            # 4. Ground contact indicators
            contact_indicators = self._calculate_ground_contact_indicators(frame_keypoints)
            frame_features.extend(contact_indicators)
            
            kinematic_features.append(frame_features)
        
        return np.array(kinematic_features)
    
    def segment_gait_cycles(self, features: np.ndarray, 
                          pose_sequence: PoseSequence) -> List[GaitCycle]:
        """
        Segment gait cycles from feature sequences.
        
        Implements gait cycle segmentation algorithms as required by the task.
        
        Args:
            features: Extracted feature array
            pose_sequence: Original pose sequence for timing information
            
        Returns:
            List of segmented gait cycles
        """
        if features.shape[0] == 0:
            return []
        
        # Method 1: Heel strike detection using ankle height
        heel_strike_cycles = self._segment_by_heel_strikes(pose_sequence)
        
        # Method 2: Peak detection in vertical ankle movement
        peak_detection_cycles = self._segment_by_peak_detection(pose_sequence)
        
        # Method 3: Frequency analysis for periodic patterns
        frequency_cycles = self._segment_by_frequency_analysis(features, pose_sequence)
        
        # Combine and validate cycles using ensemble approach
        final_cycles = self._combine_segmentation_methods(
            heel_strike_cycles, peak_detection_cycles, frequency_cycles, features
        )
        
        self.logger.info(f"Segmented {len(final_cycles)} gait cycles")
        return final_cycles
    
    def _segment_by_heel_strikes(self, pose_sequence: PoseSequence) -> List[GaitCycle]:
        """Segment gait cycles based on heel strike detection."""
        cycles = []
        
        # Extract ankle heights for both feet
        left_ankle_heights = []
        right_ankle_heights = []
        
        for frame_keypoints in pose_sequence.keypoints:
            left_ankle = frame_keypoints[self.landmark_indices['left_ankle']]
            right_ankle = frame_keypoints[self.landmark_indices['right_ankle']]
            
            left_ankle_heights.append(left_ankle.y if left_ankle.confidence > 0.3 else np.nan)
            right_ankle_heights.append(right_ankle.y if right_ankle.confidence > 0.3 else np.nan)
        
        # Find heel strikes (local minima in ankle height)
        left_strikes = self._find_heel_strikes(left_ankle_heights)
        right_strikes = self._find_heel_strikes(right_ankle_heights)
        
        # Create cycles from consecutive heel strikes
        all_strikes = sorted(left_strikes + right_strikes)
        
        for i in range(len(all_strikes) - 1):
            start_frame = all_strikes[i]
            end_frame = all_strikes[i + 1]
            
            # Validate cycle duration
            cycle_duration = (end_frame - start_frame) / self.fps
            if self.min_cycle_duration <= cycle_duration <= self.max_cycle_duration:
                cycles.append(GaitCycle(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    features=np.array([]),  # Will be filled later
                    cycle_time=cycle_duration,
                    step_count=1
                ))
        
        return cycles
    
    def _find_heel_strikes(self, ankle_heights: List[float]) -> List[int]:
        """Find heel strike events from ankle height data."""
        # Convert to numpy array and handle NaN values
        heights = np.array(ankle_heights)
        valid_mask = ~np.isnan(heights)
        
        if np.sum(valid_mask) < 10:  # Need minimum data points
            return []
        
        # Interpolate missing values
        valid_indices = np.where(valid_mask)[0]
        valid_heights = heights[valid_mask]
        
        if len(valid_indices) < 10:
            return []
        
        # Smooth the signal
        from scipy.ndimage import gaussian_filter1d
        smoothed_heights = gaussian_filter1d(valid_heights, sigma=2.0)
        
        # Find local minima (heel strikes)
        from scipy.signal import find_peaks
        
        # Invert signal to find minima as peaks
        inverted_heights = -smoothed_heights
        peaks, properties = find_peaks(
            inverted_heights,
            height=None,
            distance=int(self.min_cycle_duration * self.fps * 0.5),  # Minimum distance between strikes
            prominence=np.std(smoothed_heights) * 0.5
        )
        
        # Map back to original indices
        heel_strikes = [valid_indices[peak] for peak in peaks if peak < len(valid_indices)]
        
        return heel_strikes
    
    def _segment_by_peak_detection(self, pose_sequence: PoseSequence) -> List[GaitCycle]:
        """Segment using peak detection in ankle movement patterns."""
        cycles = []
        
        # Calculate ankle movement magnitude
        movement_signal = []
        
        for i in range(1, len(pose_sequence.keypoints)):
            prev_frame = pose_sequence.keypoints[i-1]
            curr_frame = pose_sequence.keypoints[i]
            
            # Calculate movement for both ankles
            left_prev = prev_frame[self.landmark_indices['left_ankle']]
            left_curr = curr_frame[self.landmark_indices['left_ankle']]
            right_prev = prev_frame[self.landmark_indices['right_ankle']]
            right_curr = curr_frame[self.landmark_indices['right_ankle']]
            
            if (left_prev.confidence > 0.3 and left_curr.confidence > 0.3 and
                right_prev.confidence > 0.3 and right_curr.confidence > 0.3):
                
                left_movement = euclidean([left_prev.x, left_prev.y], [left_curr.x, left_curr.y])
                right_movement = euclidean([right_prev.x, right_prev.y], [right_curr.x, right_curr.y])
                total_movement = left_movement + right_movement
            else:
                total_movement = 0.0
            
            movement_signal.append(total_movement)
        
        if len(movement_signal) < 10:
            return cycles
        
        # Find peaks in movement signal
        from scipy.signal import find_peaks
        
        peaks, _ = find_peaks(
            movement_signal,
            distance=int(self.min_cycle_duration * self.fps),
            prominence=np.std(movement_signal) * 0.3
        )
        
        # Create cycles between peaks
        for i in range(len(peaks) - 1):
            start_frame = peaks[i]
            end_frame = peaks[i + 1]
            cycle_duration = (end_frame - start_frame) / self.fps
            
            if self.min_cycle_duration <= cycle_duration <= self.max_cycle_duration:
                cycles.append(GaitCycle(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    features=np.array([]),
                    cycle_time=cycle_duration,
                    step_count=1
                ))
        
        return cycles
    
    def _segment_by_frequency_analysis(self, features: np.ndarray, 
                                     pose_sequence: PoseSequence) -> List[GaitCycle]:
        """Segment using frequency analysis of gait patterns."""
        cycles = []
        
        if features.shape[0] < 60:  # Need at least 2 seconds of data
            return cycles
        
        # Use ankle height variation for frequency analysis
        ankle_signal = []
        
        for frame_keypoints in pose_sequence.keypoints:
            left_ankle = frame_keypoints[self.landmark_indices['left_ankle']]
            right_ankle = frame_keypoints[self.landmark_indices['right_ankle']]
            
            if left_ankle.confidence > 0.3 and right_ankle.confidence > 0.3:
                # Use difference in ankle heights as signal
                height_diff = abs(left_ankle.y - right_ankle.y)
                ankle_signal.append(height_diff)
            else:
                ankle_signal.append(0.0)
        
        if len(ankle_signal) < 60:
            return cycles
        
        # Apply FFT to find dominant frequency
        from scipy.fft import fft, fftfreq
        
        signal_array = np.array(ankle_signal)
        fft_values = fft(signal_array)
        frequencies = fftfreq(len(signal_array), 1/self.fps)
        
        # Find dominant frequency in expected gait range (0.5-2.5 Hz)
        valid_freq_mask = (frequencies > 0.5) & (frequencies < 2.5)
        if np.any(valid_freq_mask):
            dominant_freq_idx = np.argmax(np.abs(fft_values[valid_freq_mask]))
            dominant_freq = frequencies[valid_freq_mask][dominant_freq_idx]
            
            # Calculate expected cycle length
            expected_cycle_length = int(self.fps / dominant_freq)
            
            # Create cycles based on expected length
            for start_idx in range(0, len(signal_array) - expected_cycle_length, expected_cycle_length):
                end_idx = start_idx + expected_cycle_length
                cycle_duration = expected_cycle_length / self.fps
                
                if self.min_cycle_duration <= cycle_duration <= self.max_cycle_duration:
                    cycles.append(GaitCycle(
                        start_frame=start_idx,
                        end_frame=end_idx,
                        features=np.array([]),
                        cycle_time=cycle_duration,
                        step_count=2  # Full gait cycle typically contains 2 steps
                    ))
        
        return cycles
    
    def _combine_segmentation_methods(self, 
                                    heel_cycles: List[GaitCycle],
                                    peak_cycles: List[GaitCycle],
                                    freq_cycles: List[GaitCycle],
                                    features: np.ndarray) -> List[GaitCycle]:
        """Combine results from different segmentation methods."""
        all_cycles = heel_cycles + peak_cycles + freq_cycles
        
        if not all_cycles:
            return []
        
        # Sort cycles by start frame
        all_cycles.sort(key=lambda x: x.start_frame)
        
        # Remove overlapping cycles (keep the one with better quality)
        final_cycles = []
        
        for cycle in all_cycles:
            # Check for overlap with existing cycles
            overlaps = False
            for existing_cycle in final_cycles:
                if (cycle.start_frame < existing_cycle.end_frame and 
                    cycle.end_frame > existing_cycle.start_frame):
                    overlaps = True
                    break
            
            if not overlaps:
                # Extract features for this cycle
                cycle_features = features[cycle.start_frame:cycle.end_frame]
                cycle.features = cycle_features
                final_cycles.append(cycle)
        
        return final_cycles
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize and standardize feature vectors.
        
        Implements feature normalization and standardization methods as required.
        
        Args:
            features: Raw feature array of shape (n_samples, n_features)
            
        Returns:
            Normalized feature array
        """
        if features.shape[0] == 0:
            return features
        
        # Handle NaN values
        features_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply standardization (zero mean, unit variance)
        try:
            features_standardized = self.spatial_scaler.fit_transform(features_clean)
        except ValueError as e:
            self.logger.warning(f"Standardization failed: {e}. Using original features.")
            features_standardized = features_clean
        
        # Apply min-max normalization to ensure values are in [0, 1] range
        try:
            features_normalized = self.temporal_scaler.fit_transform(features_standardized)
        except ValueError as e:
            self.logger.warning(f"Min-max normalization failed: {e}. Using standardized features.")
            features_normalized = features_standardized
        
        self.logger.info(f"Normalized features from shape {features.shape} to {features_normalized.shape}")
        return features_normalized
    
    # Helper methods for feature extraction
    
    def _get_normalized_joint_positions(self, frame_keypoints: List[PoseKeypoint]) -> List[float]:
        """Get joint positions normalized by torso length."""
        positions = []
        
        # Calculate torso length for normalization
        left_shoulder = frame_keypoints[self.landmark_indices['left_shoulder']]
        right_shoulder = frame_keypoints[self.landmark_indices['right_shoulder']]
        left_hip = frame_keypoints[self.landmark_indices['left_hip']]
        right_hip = frame_keypoints[self.landmark_indices['right_hip']]
        
        if (left_shoulder.confidence > 0.3 and right_shoulder.confidence > 0.3 and
            left_hip.confidence > 0.3 and right_hip.confidence > 0.3):
            
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            torso_length = abs(shoulder_center_y - hip_center_y)
            
            if torso_length > 0:
                # Normalize key joint positions
                key_joints = ['left_ankle', 'right_ankle', 'left_knee', 'right_knee']
                for joint_name in key_joints:
                    joint = frame_keypoints[self.landmark_indices[joint_name]]
                    if joint.confidence > 0.3:
                        positions.extend([
                            joint.x / torso_length,
                            joint.y / torso_length,
                            joint.z / torso_length
                        ])
                    else:
                        positions.extend([0.0, 0.0, 0.0])
            else:
                positions.extend([0.0] * 12)  # 4 joints * 3 coordinates
        else:
            positions.extend([0.0] * 12)
        
        return positions
    
    def _calculate_joint_angles(self, frame_keypoints: List[PoseKeypoint]) -> List[float]:
        """Calculate joint angles for key joints."""
        angles = []
        
        # Calculate knee angles
        left_knee_angle = self._calculate_knee_angle(frame_keypoints, 'left')
        right_knee_angle = self._calculate_knee_angle(frame_keypoints, 'right')
        
        # Calculate hip angles
        left_hip_angle = self._calculate_hip_angle(frame_keypoints, 'left')
        right_hip_angle = self._calculate_hip_angle(frame_keypoints, 'right')
        
        angles.extend([left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle])
        
        return angles
    
    def _calculate_knee_angle(self, frame_keypoints: List[PoseKeypoint], side: str) -> float:
        """Calculate knee angle for specified side."""
        hip_idx = self.landmark_indices[f'{side}_hip']
        knee_idx = self.landmark_indices[f'{side}_knee']
        ankle_idx = self.landmark_indices[f'{side}_ankle']
        
        hip = frame_keypoints[hip_idx]
        knee = frame_keypoints[knee_idx]
        ankle = frame_keypoints[ankle_idx]
        
        if hip.confidence > 0.3 and knee.confidence > 0.3 and ankle.confidence > 0.3:
            # Calculate vectors
            vec1 = np.array([hip.x - knee.x, hip.y - knee.y])
            vec2 = np.array([ankle.x - knee.x, ankle.y - knee.y])
            
            # Calculate angle
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            return float(angle)
        
        return 0.0
    
    def _calculate_hip_angle(self, frame_keypoints: List[PoseKeypoint], side: str) -> float:
        """Calculate hip angle for specified side."""
        shoulder_idx = self.landmark_indices[f'{side}_shoulder']
        hip_idx = self.landmark_indices[f'{side}_hip']
        knee_idx = self.landmark_indices[f'{side}_knee']
        
        shoulder = frame_keypoints[shoulder_idx]
        hip = frame_keypoints[hip_idx]
        knee = frame_keypoints[knee_idx]
        
        if shoulder.confidence > 0.3 and hip.confidence > 0.3 and knee.confidence > 0.3:
            # Calculate vectors
            vec1 = np.array([shoulder.x - hip.x, shoulder.y - hip.y])
            vec2 = np.array([knee.x - hip.x, knee.y - hip.y])
            
            # Calculate angle
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            return float(angle)
        
        return 0.0
    
    def _calculate_limb_lengths(self, frame_keypoints: List[PoseKeypoint]) -> List[float]:
        """Calculate limb lengths."""
        lengths = []
        
        # Calculate thigh lengths
        left_thigh = self._calculate_segment_length(
            frame_keypoints, 'left_hip', 'left_knee'
        )
        right_thigh = self._calculate_segment_length(
            frame_keypoints, 'right_hip', 'right_knee'
        )
        
        # Calculate shin lengths
        left_shin = self._calculate_segment_length(
            frame_keypoints, 'left_knee', 'left_ankle'
        )
        right_shin = self._calculate_segment_length(
            frame_keypoints, 'right_knee', 'right_ankle'
        )
        
        lengths.extend([left_thigh, right_thigh, left_shin, right_shin])
        
        return lengths
    
    def _calculate_segment_length(self, frame_keypoints: List[PoseKeypoint], 
                                joint1: str, joint2: str) -> float:
        """Calculate length between two joints."""
        idx1 = self.landmark_indices[joint1]
        idx2 = self.landmark_indices[joint2]
        
        kp1 = frame_keypoints[idx1]
        kp2 = frame_keypoints[idx2]
        
        if kp1.confidence > 0.3 and kp2.confidence > 0.3:
            return euclidean([kp1.x, kp1.y, kp1.z], [kp2.x, kp2.y, kp2.z])
        
        return 0.0
    
    def _calculate_body_symmetry(self, frame_keypoints: List[PoseKeypoint]) -> List[float]:
        """Calculate body symmetry measures."""
        symmetry_measures = []
        
        # Hip symmetry
        left_hip = frame_keypoints[self.landmark_indices['left_hip']]
        right_hip = frame_keypoints[self.landmark_indices['right_hip']]
        
        if left_hip.confidence > 0.3 and right_hip.confidence > 0.3:
            hip_height_diff = abs(left_hip.y - right_hip.y)
            symmetry_measures.append(hip_height_diff)
        else:
            symmetry_measures.append(0.0)
        
        # Shoulder symmetry
        left_shoulder = frame_keypoints[self.landmark_indices['left_shoulder']]
        right_shoulder = frame_keypoints[self.landmark_indices['right_shoulder']]
        
        if left_shoulder.confidence > 0.3 and right_shoulder.confidence > 0.3:
            shoulder_height_diff = abs(left_shoulder.y - right_shoulder.y)
            symmetry_measures.append(shoulder_height_diff)
        else:
            symmetry_measures.append(0.0)
        
        return symmetry_measures
    
    def _calculate_joint_velocities(self, pose_sequence: PoseSequence, 
                                  frame_idx: int, joints: List[str]) -> List[float]:
        """Calculate joint velocities."""
        velocities = []
        
        if frame_idx == 0:
            return [0.0] * (len(joints) * 3)  # 3D velocities
        
        prev_frame = pose_sequence.keypoints[frame_idx - 1]
        curr_frame = pose_sequence.keypoints[frame_idx]
        dt = 1.0 / self.fps
        
        for joint_name in joints:
            joint_idx = self.landmark_indices[joint_name]
            prev_joint = prev_frame[joint_idx]
            curr_joint = curr_frame[joint_idx]
            
            if prev_joint.confidence > 0.3 and curr_joint.confidence > 0.3:
                vx = (curr_joint.x - prev_joint.x) / dt
                vy = (curr_joint.y - prev_joint.y) / dt
                vz = (curr_joint.z - prev_joint.z) / dt
                velocities.extend([vx, vy, vz])
            else:
                velocities.extend([0.0, 0.0, 0.0])
        
        return velocities
    
    def _calculate_joint_accelerations(self, pose_sequence: PoseSequence,
                                     frame_idx: int, joints: List[str]) -> List[float]:
        """Calculate joint accelerations."""
        accelerations = []
        
        if frame_idx < 2:
            return [0.0] * (len(joints) * 3)  # 3D accelerations
        
        prev_prev_frame = pose_sequence.keypoints[frame_idx - 2]
        prev_frame = pose_sequence.keypoints[frame_idx - 1]
        curr_frame = pose_sequence.keypoints[frame_idx]
        dt = 1.0 / self.fps
        
        for joint_name in joints:
            joint_idx = self.landmark_indices[joint_name]
            pp_joint = prev_prev_frame[joint_idx]
            p_joint = prev_frame[joint_idx]
            c_joint = curr_frame[joint_idx]
            
            if (pp_joint.confidence > 0.3 and p_joint.confidence > 0.3 and 
                c_joint.confidence > 0.3):
                
                # Calculate velocities
                v1x = (p_joint.x - pp_joint.x) / dt
                v1y = (p_joint.y - pp_joint.y) / dt
                v1z = (p_joint.z - pp_joint.z) / dt
                
                v2x = (c_joint.x - p_joint.x) / dt
                v2y = (c_joint.y - p_joint.y) / dt
                v2z = (c_joint.z - p_joint.z) / dt
                
                # Calculate accelerations
                ax = (v2x - v1x) / dt
                ay = (v2y - v1y) / dt
                az = (v2z - v1z) / dt
                
                accelerations.extend([ax, ay, az])
            else:
                accelerations.extend([0.0, 0.0, 0.0])
        
        return accelerations
    
    def _calculate_temporal_consistency_features(self, pose_sequence: PoseSequence,
                                               frame_idx: int) -> List[float]:
        """Calculate temporal consistency features."""
        if frame_idx == 0:
            return [1.0]  # Perfect consistency for first frame
        
        prev_frame = pose_sequence.keypoints[frame_idx - 1]
        curr_frame = pose_sequence.keypoints[frame_idx]
        
        # Calculate average displacement of key joints
        key_joints = ['left_ankle', 'right_ankle', 'left_knee', 'right_knee']
        total_displacement = 0.0
        valid_joints = 0
        
        for joint_name in key_joints:
            joint_idx = self.landmark_indices[joint_name]
            prev_joint = prev_frame[joint_idx]
            curr_joint = curr_frame[joint_idx]
            
            if prev_joint.confidence > 0.3 and curr_joint.confidence > 0.3:
                displacement = euclidean(
                    [prev_joint.x, prev_joint.y],
                    [curr_joint.x, curr_joint.y]
                )
                total_displacement += displacement
                valid_joints += 1
        
        if valid_joints > 0:
            avg_displacement = total_displacement / valid_joints
            # Convert to consistency score (lower displacement = higher consistency)
            consistency = 1.0 / (1.0 + avg_displacement / 10.0)
        else:
            consistency = 0.0
        
        return [consistency]
    
    def _calculate_step_width(self, frame_keypoints: List[PoseKeypoint]) -> float:
        """Calculate step width (distance between feet)."""
        left_ankle = frame_keypoints[self.landmark_indices['left_ankle']]
        right_ankle = frame_keypoints[self.landmark_indices['right_ankle']]
        
        if left_ankle.confidence > 0.3 and right_ankle.confidence > 0.3:
            return euclidean([left_ankle.x, left_ankle.y], [right_ankle.x, right_ankle.y])
        
        return 0.0
    
    def _calculate_stride_indicators(self, pose_sequence: PoseSequence, 
                                   frame_idx: int) -> List[float]:
        """Calculate stride length indicators."""
        if frame_idx < 5:  # Need some history
            return [0.0, 0.0]  # Left and right stride indicators
        
        # Look back 5 frames to estimate stride progress
        lookback = min(5, frame_idx)
        
        left_ankle_positions = []
        right_ankle_positions = []
        
        for i in range(frame_idx - lookback, frame_idx + 1):
            frame_keypoints = pose_sequence.keypoints[i]
            left_ankle = frame_keypoints[self.landmark_indices['left_ankle']]
            right_ankle = frame_keypoints[self.landmark_indices['right_ankle']]
            
            if left_ankle.confidence > 0.3:
                left_ankle_positions.append([left_ankle.x, left_ankle.y])
            if right_ankle.confidence > 0.3:
                right_ankle_positions.append([right_ankle.x, right_ankle.y])
        
        # Calculate movement indicators
        left_stride_indicator = 0.0
        right_stride_indicator = 0.0
        
        if len(left_ankle_positions) >= 2:
            left_total_movement = sum(
                euclidean(left_ankle_positions[i], left_ankle_positions[i+1])
                for i in range(len(left_ankle_positions) - 1)
            )
            left_stride_indicator = left_total_movement
        
        if len(right_ankle_positions) >= 2:
            right_total_movement = sum(
                euclidean(right_ankle_positions[i], right_ankle_positions[i+1])
                for i in range(len(right_ankle_positions) - 1)
            )
            right_stride_indicator = right_total_movement
        
        return [left_stride_indicator, right_stride_indicator]
    
    def _calculate_center_of_mass_features(self, frame_keypoints: List[PoseKeypoint]) -> List[float]:
        """Calculate center of mass estimation features."""
        # Simplified COM calculation using key body points
        key_points = ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']
        
        valid_points = []
        for point_name in key_points:
            point = frame_keypoints[self.landmark_indices[point_name]]
            if point.confidence > 0.3:
                valid_points.append([point.x, point.y, point.z])
        
        if len(valid_points) >= 2:
            com = np.mean(valid_points, axis=0)
            return com.tolist()
        
        return [0.0, 0.0, 0.0]
    
    def _calculate_ground_contact_indicators(self, frame_keypoints: List[PoseKeypoint]) -> List[float]:
        """Calculate ground contact indicators for both feet."""
        indicators = []
        
        # For each foot, estimate ground contact based on ankle height and foot orientation
        for side in ['left', 'right']:
            ankle = frame_keypoints[self.landmark_indices[f'{side}_ankle']]
            heel = frame_keypoints[self.landmark_indices[f'{side}_heel']]
            foot_index = frame_keypoints[self.landmark_indices[f'{side}_foot_index']]
            
            if (ankle.confidence > 0.3 and heel.confidence > 0.3 and 
                foot_index.confidence > 0.3):
                
                # Ground contact indicator based on foot flatness and low height
                foot_flatness = abs(heel.y - foot_index.y)  # Lower values = flatter foot
                ankle_height = ankle.y  # Lower values = closer to ground
                
                # Combine indicators (lower values suggest ground contact)
                contact_indicator = 1.0 / (1.0 + foot_flatness + ankle_height / 100.0)
                indicators.append(contact_indicator)
            else:
                indicators.append(0.0)
        
        return indicators