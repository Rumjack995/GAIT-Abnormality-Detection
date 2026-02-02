"""
Property-based tests for comprehensive reporting functionality.

Tests Property 9: Comprehensive Report Generation and 
Property 10: Multi-Abnormality Correlation as specified in the design document.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
import numpy as np
from typing import List

from gait_analysis.analysis import AnalysisEngine, ClinicalInsightsGenerator
from gait_analysis.utils.data_structures import (
    ClassificationResult,
    GaitParameters,
    AsymmetryMetrics,
    TrackedPose,
    PoseSequence,
    PoseKeypoint
)


def create_simple_pose_sequence(num_frames=20):
    """Create a simple pose sequence for testing."""
    keypoints = []
    for frame in range(num_frames):
        frame_keypoints = []
        for i in range(33):  # MediaPipe has 33 landmarks
            x = 320 + np.sin(frame * 0.1) * 20
            y = 240 + np.cos(frame * 0.1) * 15
            z = 0.0
            confidence = 0.8
            frame_keypoints.append(PoseKeypoint(x=x, y=y, z=z, confidence=confidence))
        keypoints.append(frame_keypoints)
    
    timestamps = [i / 30.0 for i in range(num_frames)]
    confidence_scores = [0.85] * num_frames
    
    return PoseSequence(
        keypoints=keypoints,
        timestamps=timestamps,
        confidence_scores=confidence_scores
    )


def create_simple_tracked_pose():
    """Create a simple tracked pose for testing."""
    pose_sequence = create_simple_pose_sequence()
    return TrackedPose(
        pose_sequence=pose_sequence,
        tracking_id=1,
        tracking_confidence=0.85
    )


class TestComprehensiveReportingProperties:
    """Property-based tests for comprehensive reporting functionality."""
    
    # Feature: gait-abnormality-detection, Property 9: Comprehensive Report Generation
    @given(
        abnormality_type=st.sampled_from(['limping', 'shuffling', 'irregular_stride', 'balance_issues']),
        confidence=st.floats(min_value=0.5, max_value=1.0),
        severity_score=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=10, deadline=5000)
    def test_comprehensive_report_generation_property(self, abnormality_type, confidence, severity_score):
        """
        **Property 9: Comprehensive Report Generation**
        
        For any completed analysis, the system should generate reports containing 
        all required components: detected abnormalities, gait parameters 
        (stride length, cadence, step width, swing time, stance phase), 
        temporal analysis, asymmetry metrics, and clinical recommendations.
        
        **Validates: Requirements 4.1, 4.2, 4.4, 4.5**
        """
        # Create analysis engine
        analysis_engine = AnalysisEngine(fps=30.0, pixel_to_meter_ratio=0.001)
        
        # Create simple test data
        classification_results = [
            ClassificationResult(
                abnormality_type=abnormality_type,
                confidence=confidence,
                severity_score=severity_score,
                affected_limbs=['left_leg']
            )
        ]
        
        tracked_pose = create_simple_tracked_pose()
        
        # Perform comprehensive analysis
        report = analysis_engine.analyze_comprehensive(classification_results, tracked_pose)
        
        # Property: Report must contain all required components
        
        # 1. Detected abnormalities (from classification results)
        assert hasattr(report, 'classification_results')
        assert report.classification_results is not None
        assert len(report.classification_results) > 0
        
        # 2. Gait parameters (stride length, cadence, step width, swing time, stance phase)
        assert hasattr(report, 'gait_parameters')
        assert report.gait_parameters is not None
        assert hasattr(report.gait_parameters, 'stride_length')
        assert hasattr(report.gait_parameters, 'cadence')
        assert hasattr(report.gait_parameters, 'step_width')
        assert hasattr(report.gait_parameters, 'swing_time')
        assert hasattr(report.gait_parameters, 'stance_time')  # stance phase
        assert hasattr(report.gait_parameters, 'double_support_time')
        
        # Verify gait parameters have valid values
        assert report.gait_parameters.stride_length > 0
        assert report.gait_parameters.cadence > 0
        assert report.gait_parameters.step_width >= 0
        assert report.gait_parameters.swing_time > 0
        assert report.gait_parameters.stance_time > 0
        assert report.gait_parameters.double_support_time >= 0
        
        # 3. Temporal analysis
        assert hasattr(report, 'temporal_analysis')
        assert report.temporal_analysis is not None
        assert hasattr(report.temporal_analysis, 'progression_patterns')
        assert hasattr(report.temporal_analysis, 'asymmetry_over_time')
        assert hasattr(report.temporal_analysis, 'gait_variability')
        assert hasattr(report.temporal_analysis, 'cycle_consistency')
        assert hasattr(report.temporal_analysis, 'temporal_trends')
        
        # 4. Asymmetry metrics
        assert hasattr(report, 'asymmetry_metrics')
        assert report.asymmetry_metrics is not None
        assert hasattr(report.asymmetry_metrics, 'left_right_stride_ratio')
        assert hasattr(report.asymmetry_metrics, 'left_right_swing_ratio')
        assert hasattr(report.asymmetry_metrics, 'left_right_stance_ratio')
        assert hasattr(report.asymmetry_metrics, 'temporal_asymmetry')
        
        # Verify asymmetry metrics are within valid ranges
        assert 0 <= report.asymmetry_metrics.left_right_stride_ratio <= 1
        assert 0 <= report.asymmetry_metrics.left_right_swing_ratio <= 1
        assert 0 <= report.asymmetry_metrics.left_right_stance_ratio <= 1
        assert 0 <= report.asymmetry_metrics.temporal_asymmetry <= 1
        
        # 5. Clinical recommendations
        assert hasattr(report, 'clinical_insights')
        assert report.clinical_insights is not None
        assert hasattr(report.clinical_insights, 'recommendations')
        assert hasattr(report.clinical_insights, 'primary_abnormalities')
        assert hasattr(report.clinical_insights, 'risk_factors')
        
        # Verify clinical insights structure
        assert isinstance(report.clinical_insights.recommendations, list)
        assert isinstance(report.clinical_insights.primary_abnormalities, list)
        assert isinstance(report.clinical_insights.risk_factors, list)
        
        # 6. Report metadata
        assert hasattr(report, 'timestamp')
        assert hasattr(report, 'confidence_score')
        assert hasattr(report, 'processing_time_ms')
        
        # Verify metadata validity
        assert report.timestamp > 0
        assert 0 <= report.confidence_score <= 1
        assert report.processing_time_ms > 0
    
    # Feature: gait-abnormality-detection, Property 10: Multi-Abnormality Correlation
    @given(
        primary_abnormality=st.sampled_from(['limping', 'shuffling', 'irregular_stride']),
        secondary_abnormality=st.sampled_from(['balance_issues', 'irregular_stride', 'shuffling']),
        primary_confidence=st.floats(min_value=0.7, max_value=1.0),
        secondary_confidence=st.floats(min_value=0.7, max_value=1.0)
    )
    @settings(max_examples=10, deadline=5000)
    def test_multi_abnormality_correlation_property(self, primary_abnormality, secondary_abnormality, 
                                                   primary_confidence, secondary_confidence):
        """
        **Property 10: Multi-Abnormality Correlation**
        
        For any analysis with multiple detected abnormalities, the system should 
        provide correlation analysis and potential underlying causes.
        
        **Validates: Requirements 4.3**
        """
        # Ensure we have different abnormalities
        assume(primary_abnormality != secondary_abnormality)
        
        # Create analysis engine
        analysis_engine = AnalysisEngine(fps=30.0, pixel_to_meter_ratio=0.001)
        
        # Create multiple abnormality results
        classification_results = [
            ClassificationResult(
                abnormality_type=primary_abnormality,
                confidence=primary_confidence,
                severity_score=0.6,
                affected_limbs=['left_leg']
            ),
            ClassificationResult(
                abnormality_type=secondary_abnormality,
                confidence=secondary_confidence,
                severity_score=0.5,
                affected_limbs=['right_leg']
            )
        ]
        
        tracked_pose = create_simple_tracked_pose()
        
        # Perform comprehensive analysis
        report = analysis_engine.analyze_comprehensive(classification_results, tracked_pose)
        
        # Property: When multiple abnormalities are detected, correlation analysis must be provided
        
        # 1. Correlation analysis should be present
        assert hasattr(report, 'correlation_analysis')
        
        # With high confidence abnormalities, correlation analysis should exist
        high_confidence_abnormalities = [
            r for r in report.classification_results 
            if r.abnormality_type != 'normal' and r.confidence > analysis_engine.confidence_threshold
        ]
        
        if len(high_confidence_abnormalities) >= 2:
            assert report.correlation_analysis is not None
            
            # 2. Correlation analysis must have required components
            assert hasattr(report.correlation_analysis, 'primary_abnormality')
            assert hasattr(report.correlation_analysis, 'secondary_abnormalities')
            assert hasattr(report.correlation_analysis, 'correlation_strength')
            assert hasattr(report.correlation_analysis, 'potential_causes')
            assert hasattr(report.correlation_analysis, 'interaction_effects')
            
            # 3. Verify correlation analysis content
            assert report.correlation_analysis.primary_abnormality is not None
            assert report.correlation_analysis.primary_abnormality != 'normal'
            assert isinstance(report.correlation_analysis.secondary_abnormalities, list)
            assert len(report.correlation_analysis.secondary_abnormalities) > 0
            
            # 4. Correlation strength should be valid
            assert 0 <= report.correlation_analysis.correlation_strength <= 1
            
            # 5. Potential causes should be provided
            assert isinstance(report.correlation_analysis.potential_causes, list)
            
            # 6. Interaction effects should be provided
            assert isinstance(report.correlation_analysis.interaction_effects, dict)
            
            # 7. Primary abnormality should be among the detected abnormalities
            detected_abnormalities = [r.abnormality_type for r in high_confidence_abnormalities]
            assert report.correlation_analysis.primary_abnormality in detected_abnormalities
            
            # 8. Secondary abnormalities should be among detected abnormalities
            for secondary in report.correlation_analysis.secondary_abnormalities:
                assert secondary in detected_abnormalities
                assert secondary != report.correlation_analysis.primary_abnormality
    
    @given(
        stride_length=st.floats(min_value=0.5, max_value=2.5),
        cadence=st.floats(min_value=50, max_value=150),
        step_width=st.floats(min_value=0.02, max_value=0.5)
    )
    @settings(max_examples=20)
    def test_gait_parameter_validity_property(self, stride_length, cadence, step_width):
        """
        Property: Gait parameters should always be within physiologically reasonable ranges
        or flagged as abnormal.
        """
        # Create analysis engine
        analysis_engine = AnalysisEngine(fps=30.0, pixel_to_meter_ratio=0.001)
        
        gait_parameters = GaitParameters(
            stride_length=stride_length,
            cadence=cadence,
            step_width=step_width,
            swing_time=0.4,
            stance_time=0.6,
            double_support_time=0.15
        )
        
        # Test parameter validity assessment
        validity_score = analysis_engine._assess_parameter_validity(gait_parameters)
        
        # Property: Validity score should be between 0 and 1
        assert 0 <= validity_score <= 1
        
        # Property: Parameters within normal ranges should have high validity
        normal_ranges = analysis_engine.NORMAL_RANGES
        
        all_normal = True
        for param_name, (min_val, max_val) in normal_ranges.items():
            param_value = getattr(gait_parameters, param_name)
            if not (min_val <= param_value <= max_val):
                all_normal = False
                break
        
        if all_normal:
            assert validity_score > 0.8  # Should be high for all normal parameters
    
    @given(
        stride_ratio=st.floats(min_value=0.0, max_value=0.5),
        swing_ratio=st.floats(min_value=0.0, max_value=0.5),
        temporal_asymmetry=st.floats(min_value=0.0, max_value=0.5)
    )
    @settings(max_examples=20)
    def test_asymmetry_classification_property(self, stride_ratio, swing_ratio, temporal_asymmetry):
        """
        Property: Asymmetry classification should be consistent with asymmetry thresholds.
        """
        # Create analysis engine
        analysis_engine = AnalysisEngine(fps=30.0, pixel_to_meter_ratio=0.001)
        
        asymmetry_metrics = AsymmetryMetrics(
            left_right_stride_ratio=stride_ratio,
            left_right_swing_ratio=swing_ratio,
            left_right_stance_ratio=swing_ratio,
            temporal_asymmetry=temporal_asymmetry
        )
        
        severity = analysis_engine._classify_asymmetry_severity(asymmetry_metrics)
        
        # Property: Severity should be one of the valid levels
        valid_severities = ['minimal', 'mild', 'moderate', 'severe']
        assert severity in valid_severities
        
        # Property: Classification should be consistent with maximum asymmetry
        max_asymmetry = max(stride_ratio, swing_ratio, temporal_asymmetry)
        
        thresholds = analysis_engine.ASYMMETRY_THRESHOLDS
        
        if max_asymmetry < thresholds['mild']:
            assert severity == 'minimal'
        elif max_asymmetry >= thresholds['severe']:
            assert severity == 'severe'
    
    @given(
        abnormality_type=st.sampled_from(['limping', 'shuffling', 'irregular_stride', 'balance_issues']),
        confidence=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=15)
    def test_clinical_insights_consistency_property(self, abnormality_type, confidence):
        """
        Property: Clinical insights should be consistent with detected abnormalities.
        """
        # Create analysis engine
        analysis_engine = AnalysisEngine(fps=30.0, pixel_to_meter_ratio=0.001)
        
        classification_results = [
            ClassificationResult(
                abnormality_type=abnormality_type,
                confidence=confidence,
                severity_score=0.5,
                affected_limbs=['left_leg']
            )
        ]
        
        # Create minimal valid gait parameters and asymmetry metrics
        gait_parameters = GaitParameters(
            stride_length=1.2, cadence=110, step_width=0.10,
            swing_time=0.4, stance_time=0.6, double_support_time=0.15
        )
        
        asymmetry_metrics = AsymmetryMetrics(
            left_right_stride_ratio=0.05, left_right_swing_ratio=0.04,
            left_right_stance_ratio=0.03, temporal_asymmetry=0.05
        )
        
        insights = analysis_engine.generate_insights(
            classification_results, gait_parameters, asymmetry_metrics
        )
        
        # Property: Primary abnormalities should match high-confidence classifications
        high_confidence_abnormalities = [
            r.abnormality_type for r in classification_results 
            if r.abnormality_type != 'normal' and r.confidence > analysis_engine.confidence_threshold
        ]
        
        for abnormality in insights.primary_abnormalities:
            assert abnormality in high_confidence_abnormalities
        
        # Property: Recommendations should be provided
        assert isinstance(insights.recommendations, list)
        
        # Property: Risk factors should be provided
        assert isinstance(insights.risk_factors, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])