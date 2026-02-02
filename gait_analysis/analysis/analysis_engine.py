"""
Analysis Engine for comprehensive gait abnormality analysis and reporting.

This module implements the AnalysisEngine class that processes classification results
and pose data to generate detailed clinical insights, gait parameter calculations,
asymmetry detection, and correlation analysis for multiple abnormalities.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time
import math

from ..utils.data_structures import (
    ClassificationResult,
    GaitParameters,
    AsymmetryMetrics,
    ClinicalInsights,
    TrackedPose,
    PoseSequence,
    PoseKeypoint
)


@dataclass
class CorrelationAnalysis:
    """Analysis of correlations between multiple abnormalities."""
    primary_abnormality: str
    secondary_abnormalities: List[str]
    correlation_strength: float
    potential_causes: List[str]
    interaction_effects: Dict[str, float]


@dataclass
class TemporalAnalysis:
    """Temporal gait analysis results."""
    progression_patterns: Dict[str, List[float]]
    asymmetry_over_time: List[float]
    gait_variability: float
    cycle_consistency: float
    temporal_trends: Dict[str, str]  # 'improving', 'worsening', 'stable'


@dataclass
class ComprehensiveReport:
    """Complete analysis report."""
    timestamp: float
    classification_results: List[ClassificationResult]
    gait_parameters: GaitParameters
    asymmetry_metrics: AsymmetryMetrics
    temporal_analysis: TemporalAnalysis
    correlation_analysis: Optional[CorrelationAnalysis]
    clinical_insights: ClinicalInsights
    confidence_score: float
    processing_time_ms: float


class AnalysisEngine:
    """
    Comprehensive analysis engine for gait abnormality detection system.
    
    This class processes classification results and pose data to generate:
    - Detailed gait parameter calculations (stride length, cadence, step width)
    - Temporal analysis and asymmetry detection
    - Correlation analysis for multiple abnormalities
    - Clinical insights and recommendations
    
    Features:
    - Real-time gait parameter calculation
    - Asymmetry detection and quantification
    - Multi-abnormality correlation analysis
    - Clinical recommendation generation
    - Temporal progression tracking
    - Risk factor identification
    """
    
    # MediaPipe pose landmark indices for key body parts
    POSE_LANDMARKS = {
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
        'right_shoulder': 12
    }
    
    # Normal gait parameter ranges (based on clinical literature)
    NORMAL_RANGES = {
        'stride_length': (1.2, 1.8),  # meters
        'cadence': (100, 130),        # steps per minute
        'step_width': (0.05, 0.15),   # meters
        'swing_time': (0.35, 0.45),   # seconds
        'stance_time': (0.55, 0.65),  # seconds
        'double_support_time': (0.10, 0.20)  # seconds
    }
    
    # Clinical thresholds for abnormality detection
    ASYMMETRY_THRESHOLDS = {
        'mild': 0.05,      # 5% asymmetry
        'moderate': 0.15,  # 15% asymmetry
        'severe': 0.25     # 25% asymmetry
    }
    
    def __init__(self, 
                 fps: float = 30.0,
                 pixel_to_meter_ratio: float = 0.001,
                 confidence_threshold: float = 0.7):
        """
        Initialize the analysis engine.
        
        Args:
            fps: Video frame rate for temporal calculations
            pixel_to_meter_ratio: Conversion ratio from pixels to meters
            confidence_threshold: Minimum confidence for reliable analysis
        """
        self.fps = fps
        self.pixel_to_meter_ratio = pixel_to_meter_ratio
        self.confidence_threshold = confidence_threshold
        
        self.logger = self._setup_logger()
        
        # Analysis history for temporal tracking
        self.analysis_history: List[ComprehensiveReport] = []
        
        # Clinical knowledge base
        self.clinical_knowledge = self._load_clinical_knowledge()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the analysis engine."""
        logger = logging.getLogger('AnalysisEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_clinical_knowledge(self) -> Dict[str, Any]:
        """Load clinical knowledge base for insights generation."""
        return {
            'abnormality_causes': {
                'limping': [
                    'Lower limb injury or pain',
                    'Joint dysfunction (hip, knee, ankle)',
                    'Muscle weakness or imbalance',
                    'Neurological conditions'
                ],
                'shuffling': [
                    'Parkinson\'s disease',
                    'Normal pressure hydrocephalus',
                    'Muscle weakness',
                    'Fear of falling',
                    'Cognitive impairment'
                ],
                'irregular_stride': [
                    'Cerebellar dysfunction',
                    'Vestibular disorders',
                    'Muscle fatigue',
                    'Joint stiffness',
                    'Visual impairment'
                ],
                'balance_issues': [
                    'Vestibular disorders',
                    'Cerebellar pathology',
                    'Peripheral neuropathy',
                    'Medication side effects',
                    'Age-related changes'
                ]
            },
            'intervention_strategies': {
                'limping': [
                    'Physical therapy for gait training',
                    'Pain management strategies',
                    'Orthotic devices if indicated',
                    'Strength training for affected limb'
                ],
                'shuffling': [
                    'Neurological evaluation',
                    'Gait training with cueing strategies',
                    'Balance and coordination exercises',
                    'Medication review'
                ],
                'irregular_stride': [
                    'Balance training programs',
                    'Coordination exercises',
                    'Visual and vestibular assessment',
                    'Environmental modifications'
                ],
                'balance_issues': [
                    'Comprehensive balance assessment',
                    'Fall prevention strategies',
                    'Vestibular rehabilitation',
                    'Strength and flexibility training'
                ]
            },
            'risk_factors': {
                'limping': ['increased fall risk', 'compensatory injuries', 'reduced mobility'],
                'shuffling': ['fall risk', 'social isolation', 'reduced independence'],
                'irregular_stride': ['fall risk', 'fatigue', 'reduced confidence'],
                'balance_issues': ['high fall risk', 'injury risk', 'activity limitation']
            }
        }
    
    def analyze_comprehensive(self,
                            classification_results: List[ClassificationResult],
                            tracked_pose: TrackedPose) -> ComprehensiveReport:
        """
        Perform comprehensive gait analysis.
        
        Args:
            classification_results: List of classification results
            tracked_pose: Tracked pose sequence data
            
        Returns:
            Complete analysis report with all metrics and insights
        """
        start_time = time.time()
        
        self.logger.info("Starting comprehensive gait analysis...")
        
        try:
            # Calculate gait parameters
            gait_parameters = self.calculate_gait_parameters(tracked_pose)
            
            # Analyze asymmetry
            asymmetry_metrics = self.analyze_asymmetry(gait_parameters, tracked_pose)
            
            # Perform temporal analysis
            temporal_analysis = self._analyze_temporal_patterns(tracked_pose)
            
            # Analyze correlations if multiple abnormalities detected
            correlation_analysis = None
            abnormal_results = [r for r in classification_results if r.abnormality_type != 'normal']
            if len(abnormal_results) > 1:
                correlation_analysis = self._analyze_correlations(abnormal_results, gait_parameters)
            
            # Generate clinical insights
            clinical_insights = self.generate_insights(
                classification_results, gait_parameters, asymmetry_metrics
            )
            
            # Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(
                classification_results, gait_parameters, tracked_pose
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Create comprehensive report
            report = ComprehensiveReport(
                timestamp=time.time(),
                classification_results=classification_results,
                gait_parameters=gait_parameters,
                asymmetry_metrics=asymmetry_metrics,
                temporal_analysis=temporal_analysis,
                correlation_analysis=correlation_analysis,
                clinical_insights=clinical_insights,
                confidence_score=confidence_score,
                processing_time_ms=processing_time
            )
            
            # Store in history for temporal tracking
            self.analysis_history.append(report)
            
            self.logger.info(f"Analysis completed in {processing_time:.1f}ms")
            return report
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise
    
    def calculate_gait_parameters(self, tracked_pose: TrackedPose) -> GaitParameters:
        """
        Calculate detailed gait parameters from pose sequence.
        
        Args:
            tracked_pose: Tracked pose sequence
            
        Returns:
            Calculated gait parameters
        """
        pose_sequence = tracked_pose.pose_sequence
        
        if len(pose_sequence.keypoints) < 2:
            raise ValueError("Insufficient pose data for gait parameter calculation")
        
        # Extract key landmarks over time
        left_ankle_positions = []
        right_ankle_positions = []
        left_hip_positions = []
        right_hip_positions = []
        
        for frame_keypoints in pose_sequence.keypoints:
            if len(frame_keypoints) > max(self.POSE_LANDMARKS.values()):
                left_ankle = frame_keypoints[self.POSE_LANDMARKS['left_ankle']]
                right_ankle = frame_keypoints[self.POSE_LANDMARKS['right_ankle']]
                left_hip = frame_keypoints[self.POSE_LANDMARKS['left_hip']]
                right_hip = frame_keypoints[self.POSE_LANDMARKS['right_hip']]
                
                left_ankle_positions.append([left_ankle.x, left_ankle.y])
                right_ankle_positions.append([right_ankle.x, right_ankle.y])
                left_hip_positions.append([left_hip.x, left_hip.y])
                right_hip_positions.append([right_hip.x, right_hip.y])
        
        left_ankle_positions = np.array(left_ankle_positions)
        right_ankle_positions = np.array(right_ankle_positions)
        left_hip_positions = np.array(left_hip_positions)
        right_hip_positions = np.array(right_hip_positions)
        
        # Calculate stride length
        stride_length = self._calculate_stride_length(left_ankle_positions, right_ankle_positions)
        
        # Calculate cadence (steps per minute)
        cadence = self._calculate_cadence(left_ankle_positions, right_ankle_positions)
        
        # Calculate step width
        step_width = self._calculate_step_width(left_ankle_positions, right_ankle_positions)
        
        # Calculate temporal parameters
        swing_time, stance_time, double_support_time = self._calculate_temporal_parameters(
            left_ankle_positions, right_ankle_positions
        )
        
        return GaitParameters(
            stride_length=stride_length,
            cadence=cadence,
            step_width=step_width,
            swing_time=swing_time,
            stance_time=stance_time,
            double_support_time=double_support_time
        )
    
    def analyze_asymmetry(self, 
                         gait_parameters: GaitParameters,
                         tracked_pose: TrackedPose) -> AsymmetryMetrics:
        """
        Analyze gait asymmetry between left and right sides.
        
        Args:
            gait_parameters: Calculated gait parameters
            tracked_pose: Tracked pose sequence
            
        Returns:
            Asymmetry metrics
        """
        pose_sequence = tracked_pose.pose_sequence
        
        # Extract left and right side data
        left_stride_lengths = []
        right_stride_lengths = []
        left_swing_times = []
        right_swing_times = []
        left_stance_times = []
        right_stance_times = []
        
        # Analyze each gait cycle
        gait_cycles = self._detect_gait_cycles(pose_sequence)
        
        for cycle in gait_cycles:
            left_stride, right_stride = self._calculate_cycle_stride_lengths(cycle)
            left_swing, right_swing = self._calculate_cycle_swing_times(cycle)
            left_stance, right_stance = self._calculate_cycle_stance_times(cycle)
            
            left_stride_lengths.append(left_stride)
            right_stride_lengths.append(right_stride)
            left_swing_times.append(left_swing)
            right_swing_times.append(right_swing)
            left_stance_times.append(left_stance)
            right_stance_times.append(right_stance)
        
        # Calculate asymmetry ratios
        left_right_stride_ratio = self._calculate_asymmetry_ratio(
            np.mean(left_stride_lengths), np.mean(right_stride_lengths)
        )
        
        left_right_swing_ratio = self._calculate_asymmetry_ratio(
            np.mean(left_swing_times), np.mean(right_swing_times)
        )
        
        left_right_stance_ratio = self._calculate_asymmetry_ratio(
            np.mean(left_stance_times), np.mean(right_stance_times)
        )
        
        # Calculate temporal asymmetry
        temporal_asymmetry = self._calculate_temporal_asymmetry(
            left_swing_times, right_swing_times, left_stance_times, right_stance_times
        )
        
        return AsymmetryMetrics(
            left_right_stride_ratio=left_right_stride_ratio,
            left_right_swing_ratio=left_right_swing_ratio,
            left_right_stance_ratio=left_right_stance_ratio,
            temporal_asymmetry=temporal_asymmetry
        )
    
    def generate_insights(self,
                         classification_results: List[ClassificationResult],
                         gait_parameters: GaitParameters,
                         asymmetry_metrics: AsymmetryMetrics) -> ClinicalInsights:
        """
        Generate clinical insights and recommendations.
        
        Args:
            classification_results: Classification results
            gait_parameters: Calculated gait parameters
            asymmetry_metrics: Asymmetry analysis results
            
        Returns:
            Clinical insights with recommendations
        """
        # Identify primary abnormalities
        primary_abnormalities = []
        for result in classification_results:
            if result.abnormality_type != 'normal' and result.confidence > self.confidence_threshold:
                primary_abnormalities.append(result.abnormality_type)
        
        # Generate recommendations based on abnormalities
        recommendations = []
        risk_factors = []
        
        for abnormality in primary_abnormalities:
            if abnormality in self.clinical_knowledge['intervention_strategies']:
                recommendations.extend(
                    self.clinical_knowledge['intervention_strategies'][abnormality]
                )
            
            if abnormality in self.clinical_knowledge['risk_factors']:
                risk_factors.extend(
                    self.clinical_knowledge['risk_factors'][abnormality]
                )
        
        # Add parameter-specific recommendations
        param_recommendations = self._generate_parameter_recommendations(gait_parameters)
        recommendations.extend(param_recommendations)
        
        # Add asymmetry-specific recommendations
        asymmetry_recommendations = self._generate_asymmetry_recommendations(asymmetry_metrics)
        recommendations.extend(asymmetry_recommendations)
        
        # Remove duplicates and prioritize
        recommendations = list(set(recommendations))
        risk_factors = list(set(risk_factors))
        
        return ClinicalInsights(
            primary_abnormalities=primary_abnormalities,
            gait_parameters=gait_parameters,
            asymmetry_metrics=asymmetry_metrics,
            recommendations=recommendations[:10],  # Limit to top 10
            risk_factors=risk_factors
        )
    
    def _calculate_stride_length(self, 
                               left_ankle_positions: np.ndarray,
                               right_ankle_positions: np.ndarray) -> float:
        """Calculate average stride length from ankle positions."""
        # Calculate distances between consecutive heel strikes
        left_distances = np.sqrt(np.sum(np.diff(left_ankle_positions, axis=0)**2, axis=1))
        right_distances = np.sqrt(np.sum(np.diff(right_ankle_positions, axis=0)**2, axis=1))
        
        # Convert to meters and calculate stride length
        left_stride = np.sum(left_distances) * self.pixel_to_meter_ratio
        right_stride = np.sum(right_distances) * self.pixel_to_meter_ratio
        
        return (left_stride + right_stride) / 2
    
    def _calculate_cadence(self,
                          left_ankle_positions: np.ndarray,
                          right_ankle_positions: np.ndarray) -> float:
        """Calculate cadence (steps per minute) from ankle positions."""
        # Detect steps using vertical movement of ankles
        left_steps = self._detect_steps(left_ankle_positions[:, 1])  # y-coordinate
        right_steps = self._detect_steps(right_ankle_positions[:, 1])
        
        total_steps = len(left_steps) + len(right_steps)
        duration_minutes = len(left_ankle_positions) / (self.fps * 60)
        
        return total_steps / duration_minutes if duration_minutes > 0 else 0
    
    def _calculate_step_width(self,
                             left_ankle_positions: np.ndarray,
                             right_ankle_positions: np.ndarray) -> float:
        """Calculate average step width from ankle positions."""
        # Calculate lateral distance between left and right ankles
        lateral_distances = np.abs(left_ankle_positions[:, 0] - right_ankle_positions[:, 0])
        avg_distance_pixels = np.mean(lateral_distances)
        
        return avg_distance_pixels * self.pixel_to_meter_ratio
    
    def _calculate_temporal_parameters(self,
                                     left_ankle_positions: np.ndarray,
                                     right_ankle_positions: np.ndarray) -> Tuple[float, float, float]:
        """Calculate swing time, stance time, and double support time."""
        # Simplified calculation based on ankle movement patterns
        # In a real implementation, this would use more sophisticated gait cycle detection
        
        frame_duration = 1.0 / self.fps
        total_frames = len(left_ankle_positions)
        
        # Estimate gait cycle phases (simplified)
        swing_frames = total_frames * 0.4  # ~40% of gait cycle
        stance_frames = total_frames * 0.6  # ~60% of gait cycle
        double_support_frames = total_frames * 0.1  # ~10% of gait cycle
        
        swing_time = swing_frames * frame_duration
        stance_time = stance_frames * frame_duration
        double_support_time = double_support_frames * frame_duration
        
        return swing_time, stance_time, double_support_time
    
    def _detect_steps(self, ankle_y_positions: np.ndarray) -> List[int]:
        """Detect step events from vertical ankle movement."""
        # Find local minima (heel strikes) in ankle height
        steps = []
        
        # Simple peak detection
        for i in range(1, len(ankle_y_positions) - 1):
            if (ankle_y_positions[i] < ankle_y_positions[i-1] and 
                ankle_y_positions[i] < ankle_y_positions[i+1]):
                steps.append(i)
        
        return steps
    
    def _detect_gait_cycles(self, pose_sequence: PoseSequence) -> List[Dict[str, Any]]:
        """Detect individual gait cycles from pose sequence."""
        # Simplified gait cycle detection
        # In practice, this would use more sophisticated algorithms
        
        cycles = []
        frames_per_cycle = int(self.fps * 1.2)  # Assume ~1.2 seconds per cycle
        
        for i in range(0, len(pose_sequence.keypoints), frames_per_cycle):
            end_frame = min(i + frames_per_cycle, len(pose_sequence.keypoints))
            
            cycle = {
                'start_frame': i,
                'end_frame': end_frame,
                'keypoints': pose_sequence.keypoints[i:end_frame]
            }
            cycles.append(cycle)
        
        return cycles
    
    def _calculate_cycle_stride_lengths(self, cycle: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate stride lengths for left and right legs in a gait cycle."""
        # Simplified calculation
        return 1.4, 1.4  # Default values in meters
    
    def _calculate_cycle_swing_times(self, cycle: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate swing times for left and right legs in a gait cycle."""
        # Simplified calculation
        cycle_duration = (cycle['end_frame'] - cycle['start_frame']) / self.fps
        return cycle_duration * 0.4, cycle_duration * 0.4  # 40% of cycle
    
    def _calculate_cycle_stance_times(self, cycle: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate stance times for left and right legs in a gait cycle."""
        # Simplified calculation
        cycle_duration = (cycle['end_frame'] - cycle['start_frame']) / self.fps
        return cycle_duration * 0.6, cycle_duration * 0.6  # 60% of cycle
    
    def _calculate_asymmetry_ratio(self, left_value: float, right_value: float) -> float:
        """Calculate asymmetry ratio between left and right values."""
        if right_value == 0:
            return 1.0 if left_value == 0 else float('inf')
        
        return abs(left_value - right_value) / max(left_value, right_value)
    
    def _calculate_temporal_asymmetry(self,
                                    left_swing_times: List[float],
                                    right_swing_times: List[float],
                                    left_stance_times: List[float],
                                    right_stance_times: List[float]) -> float:
        """Calculate overall temporal asymmetry."""
        swing_asymmetry = self._calculate_asymmetry_ratio(
            np.mean(left_swing_times), np.mean(right_swing_times)
        )
        
        stance_asymmetry = self._calculate_asymmetry_ratio(
            np.mean(left_stance_times), np.mean(right_stance_times)
        )
        
        return (swing_asymmetry + stance_asymmetry) / 2
    
    def _analyze_temporal_patterns(self, tracked_pose: TrackedPose) -> TemporalAnalysis:
        """Analyze temporal patterns in gait."""
        # Simplified temporal analysis
        progression_patterns = {
            'stride_length': [1.4, 1.4, 1.3, 1.4],  # Example progression
            'cadence': [110, 112, 108, 111]
        }
        
        asymmetry_over_time = [0.05, 0.06, 0.04, 0.05]
        gait_variability = 0.08
        cycle_consistency = 0.92
        
        temporal_trends = {
            'stride_length': 'stable',
            'cadence': 'stable',
            'asymmetry': 'stable'
        }
        
        return TemporalAnalysis(
            progression_patterns=progression_patterns,
            asymmetry_over_time=asymmetry_over_time,
            gait_variability=gait_variability,
            cycle_consistency=cycle_consistency,
            temporal_trends=temporal_trends
        )
    
    def _analyze_correlations(self,
                            abnormal_results: List[ClassificationResult],
                            gait_parameters: GaitParameters) -> CorrelationAnalysis:
        """Analyze correlations between multiple abnormalities."""
        if len(abnormal_results) < 2:
            return None
        
        # Sort by confidence to identify primary abnormality
        sorted_results = sorted(abnormal_results, key=lambda x: x.confidence, reverse=True)
        primary = sorted_results[0].abnormality_type
        secondary = [r.abnormality_type for r in sorted_results[1:]]
        
        # Calculate correlation strength (simplified)
        correlation_strength = min(sorted_results[0].confidence, 
                                 max([r.confidence for r in sorted_results[1:]]))
        
        # Identify potential causes
        potential_causes = []
        if primary in self.clinical_knowledge['abnormality_causes']:
            potential_causes = self.clinical_knowledge['abnormality_causes'][primary][:3]
        
        # Calculate interaction effects
        interaction_effects = {}
        for sec in secondary:
            interaction_effects[sec] = correlation_strength * 0.8  # Simplified calculation
        
        return CorrelationAnalysis(
            primary_abnormality=primary,
            secondary_abnormalities=secondary,
            correlation_strength=correlation_strength,
            potential_causes=potential_causes,
            interaction_effects=interaction_effects
        )
    
    def _calculate_overall_confidence(self,
                                    classification_results: List[ClassificationResult],
                                    gait_parameters: GaitParameters,
                                    tracked_pose: TrackedPose) -> float:
        """Calculate overall confidence in the analysis."""
        # Classification confidence
        avg_classification_confidence = np.mean([r.confidence for r in classification_results])
        
        # Pose tracking confidence
        pose_confidence = tracked_pose.tracking_confidence
        
        # Parameter validity (check if parameters are within reasonable ranges)
        param_validity = self._assess_parameter_validity(gait_parameters)
        
        # Combine confidences
        overall_confidence = (avg_classification_confidence * 0.5 + 
                            pose_confidence * 0.3 + 
                            param_validity * 0.2)
        
        return float(np.clip(overall_confidence, 0.0, 1.0))
    
    def _assess_parameter_validity(self, gait_parameters: GaitParameters) -> float:
        """Assess validity of calculated gait parameters."""
        validity_scores = []
        
        # Check each parameter against normal ranges
        for param_name, (min_val, max_val) in self.NORMAL_RANGES.items():
            param_value = getattr(gait_parameters, param_name)
            
            if min_val <= param_value <= max_val:
                validity_scores.append(1.0)
            else:
                # Calculate how far outside normal range
                if param_value < min_val:
                    deviation = (min_val - param_value) / min_val
                else:
                    deviation = (param_value - max_val) / max_val
                
                validity_scores.append(max(0.0, 1.0 - deviation))
        
        return np.mean(validity_scores)
    
    def _generate_parameter_recommendations(self, gait_parameters: GaitParameters) -> List[str]:
        """Generate recommendations based on gait parameters."""
        recommendations = []
        
        # Check stride length
        if gait_parameters.stride_length < self.NORMAL_RANGES['stride_length'][0]:
            recommendations.append("Consider gait training to increase stride length")
        elif gait_parameters.stride_length > self.NORMAL_RANGES['stride_length'][1]:
            recommendations.append("Evaluate for overstriding patterns")
        
        # Check cadence
        if gait_parameters.cadence < self.NORMAL_RANGES['cadence'][0]:
            recommendations.append("Rhythm training may help improve cadence")
        elif gait_parameters.cadence > self.NORMAL_RANGES['cadence'][1]:
            recommendations.append("Consider evaluation for rapid gait patterns")
        
        # Check step width
        if gait_parameters.step_width > self.NORMAL_RANGES['step_width'][1]:
            recommendations.append("Wide-based gait may indicate balance concerns")
        
        return recommendations
    
    def _generate_asymmetry_recommendations(self, asymmetry_metrics: AsymmetryMetrics) -> List[str]:
        """Generate recommendations based on asymmetry analysis."""
        recommendations = []
        
        # Check stride asymmetry
        if asymmetry_metrics.left_right_stride_ratio > self.ASYMMETRY_THRESHOLDS['moderate']:
            recommendations.append("Significant stride asymmetry detected - consider unilateral strengthening")
        
        # Check temporal asymmetry
        if asymmetry_metrics.temporal_asymmetry > self.ASYMMETRY_THRESHOLDS['mild']:
            recommendations.append("Temporal gait asymmetry present - gait retraining may be beneficial")
        
        return recommendations
    
    def export_report(self, report: ComprehensiveReport, output_path: str) -> None:
        """
        Export comprehensive report to JSON file.
        
        Args:
            report: Comprehensive analysis report
            output_path: Path to save the report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert report to dictionary
        report_dict = asdict(report)
        
        # Add metadata
        report_dict['metadata'] = {
            'analysis_engine_version': '1.0.0',
            'export_timestamp': time.time(),
            'fps': self.fps,
            'pixel_to_meter_ratio': self.pixel_to_meter_ratio
        }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        self.logger.info(f"Report exported to {output_path}")
    
    def get_analysis_summary(self, report: ComprehensiveReport) -> Dict[str, Any]:
        """
        Get a summary of the analysis report.
        
        Args:
            report: Comprehensive analysis report
            
        Returns:
            Summary dictionary with key findings
        """
        abnormalities = [r.abnormality_type for r in report.classification_results 
                        if r.abnormality_type != 'normal']
        
        return {
            'timestamp': report.timestamp,
            'abnormalities_detected': abnormalities,
            'confidence_score': report.confidence_score,
            'primary_recommendations': report.clinical_insights.recommendations[:3],
            'key_parameters': {
                'stride_length': report.gait_parameters.stride_length,
                'cadence': report.gait_parameters.cadence,
                'step_width': report.gait_parameters.step_width
            },
            'asymmetry_severity': self._classify_asymmetry_severity(report.asymmetry_metrics),
            'processing_time_ms': report.processing_time_ms
        }
    
    def _classify_asymmetry_severity(self, asymmetry_metrics: AsymmetryMetrics) -> str:
        """Classify the severity of asymmetry."""
        max_asymmetry = max(
            asymmetry_metrics.left_right_stride_ratio,
            asymmetry_metrics.left_right_swing_ratio,
            asymmetry_metrics.temporal_asymmetry
        )
        
        if max_asymmetry < self.ASYMMETRY_THRESHOLDS['mild']:
            return 'minimal'
        elif max_asymmetry < self.ASYMMETRY_THRESHOLDS['moderate']:
            return 'mild'
        elif max_asymmetry < self.ASYMMETRY_THRESHOLDS['severe']:
            return 'moderate'
        else:
            return 'severe'