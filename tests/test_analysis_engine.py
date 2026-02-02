"""
Unit tests for the AnalysisEngine class.

Tests gait parameter calculation, asymmetry detection, correlation analysis,
and clinical insights generation functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import time

from gait_analysis.analysis import (
    AnalysisEngine, 
    ComprehensiveReport,
    ClinicalInsightsGenerator,
    SeverityLevel
)
from gait_analysis.utils.data_structures import (
    ClassificationResult,
    GaitParameters,
    AsymmetryMetrics,
    TrackedPose,
    PoseSequence,
    PoseKeypoint,
    ClinicalInsights
)


class TestAnalysisEngine:
    """Test suite for AnalysisEngine class."""
    
    @pytest.fixture
    def analysis_engine(self):
        """Create AnalysisEngine instance for testing."""
        return AnalysisEngine(fps=30.0, pixel_to_meter_ratio=0.001)
    
    @pytest.fixture
    def sample_pose_keypoints(self):
        """Create sample pose keypoints for testing."""
        keypoints = []
        for frame in range(60):  # 2 seconds at 30 fps
            frame_keypoints = []
            for i in range(33):  # MediaPipe has 33 landmarks
                # Create realistic keypoint positions with some variation
                x = 320 + np.sin(frame * 0.1) * 50 + np.random.normal(0, 5)
                y = 240 + np.cos(frame * 0.1) * 30 + np.random.normal(0, 3)
                z = np.random.normal(0, 0.1)
                confidence = 0.8 + np.random.normal(0, 0.1)
                
                frame_keypoints.append(PoseKeypoint(x=x, y=y, z=z, confidence=confidence))
            keypoints.append(frame_keypoints)
        
        return keypoints
    
    @pytest.fixture
    def sample_pose_sequence(self, sample_pose_keypoints):
        """Create sample pose sequence for testing."""
        timestamps = [i / 30.0 for i in range(len(sample_pose_keypoints))]
        confidence_scores = [0.85] * len(sample_pose_keypoints)
        
        return PoseSequence(
            keypoints=sample_pose_keypoints,
            timestamps=timestamps,
            confidence_scores=confidence_scores
        )
    
    @pytest.fixture
    def sample_tracked_pose(self, sample_pose_sequence):
        """Create sample tracked pose for testing."""
        return TrackedPose(
            pose_sequence=sample_pose_sequence,
            tracking_id=1,
            tracking_confidence=0.85
        )
    
    @pytest.fixture
    def sample_classification_results(self):
        """Create sample classification results for testing."""
        return [
            ClassificationResult(
                abnormality_type='limping',
                confidence=0.85,
                severity_score=0.6,
                affected_limbs=['left_leg']
            ),
            ClassificationResult(
                abnormality_type='normal',
                confidence=0.15,
                severity_score=0.1,
                affected_limbs=[]
            )
        ]
    
    def test_initialization(self, analysis_engine):
        """Test AnalysisEngine initialization."""
        assert analysis_engine.fps == 30.0
        assert analysis_engine.pixel_to_meter_ratio == 0.001
        assert analysis_engine.confidence_threshold == 0.7
        assert len(analysis_engine.analysis_history) == 0
        assert analysis_engine.clinical_knowledge is not None
    
    def test_calculate_gait_parameters(self, analysis_engine, sample_tracked_pose):
        """Test gait parameter calculation."""
        gait_parameters = analysis_engine.calculate_gait_parameters(sample_tracked_pose)
        
        # Verify return type
        assert isinstance(gait_parameters, GaitParameters)
        
        # Verify all parameters are calculated
        assert hasattr(gait_parameters, 'stride_length')
        assert hasattr(gait_parameters, 'cadence')
        assert hasattr(gait_parameters, 'step_width')
        assert hasattr(gait_parameters, 'swing_time')
        assert hasattr(gait_parameters, 'stance_time')
        assert hasattr(gait_parameters, 'double_support_time')
        
        # Verify reasonable values
        assert gait_parameters.stride_length > 0
        assert gait_parameters.cadence > 0
        assert gait_parameters.step_width >= 0
        assert gait_parameters.swing_time > 0
        assert gait_parameters.stance_time > 0
        assert gait_parameters.double_support_time >= 0
    
    def test_calculate_gait_parameters_insufficient_data(self, analysis_engine):
        """Test gait parameter calculation with insufficient data."""
        # Create pose sequence with only one frame
        single_keypoint = [[PoseKeypoint(x=0, y=0, z=0, confidence=0.8) for _ in range(33)]]
        pose_sequence = PoseSequence(
            keypoints=single_keypoint,
            timestamps=[0.0],
            confidence_scores=[0.8]
        )
        tracked_pose = TrackedPose(
            pose_sequence=pose_sequence,
            tracking_id=1,
            tracking_confidence=0.8
        )
        
        with pytest.raises(ValueError, match="Insufficient pose data"):
            analysis_engine.calculate_gait_parameters(tracked_pose)
    
    def test_analyze_asymmetry(self, analysis_engine, sample_tracked_pose):
        """Test asymmetry analysis."""
        # First calculate gait parameters
        gait_parameters = analysis_engine.calculate_gait_parameters(sample_tracked_pose)
        
        # Then analyze asymmetry
        asymmetry_metrics = analysis_engine.analyze_asymmetry(gait_parameters, sample_tracked_pose)
        
        # Verify return type
        assert isinstance(asymmetry_metrics, AsymmetryMetrics)
        
        # Verify all metrics are calculated
        assert hasattr(asymmetry_metrics, 'left_right_stride_ratio')
        assert hasattr(asymmetry_metrics, 'left_right_swing_ratio')
        assert hasattr(asymmetry_metrics, 'left_right_stance_ratio')
        assert hasattr(asymmetry_metrics, 'temporal_asymmetry')
        
        # Verify reasonable values (asymmetry ratios should be between 0 and 1)
        assert 0 <= asymmetry_metrics.left_right_stride_ratio <= 1
        assert 0 <= asymmetry_metrics.left_right_swing_ratio <= 1
        assert 0 <= asymmetry_metrics.left_right_stance_ratio <= 1
        assert 0 <= asymmetry_metrics.temporal_asymmetry <= 1
    
    def test_generate_insights(self, analysis_engine, sample_classification_results):
        """Test clinical insights generation."""
        # Create sample gait parameters and asymmetry metrics
        gait_parameters = GaitParameters(
            stride_length=1.2,
            cadence=110,
            step_width=0.12,
            swing_time=0.4,
            stance_time=0.6,
            double_support_time=0.15
        )
        
        asymmetry_metrics = AsymmetryMetrics(
            left_right_stride_ratio=0.08,
            left_right_swing_ratio=0.06,
            left_right_stance_ratio=0.05,
            temporal_asymmetry=0.07
        )
        
        insights = analysis_engine.generate_insights(
            sample_classification_results, gait_parameters, asymmetry_metrics
        )
        
        # Verify return type
        assert isinstance(insights, ClinicalInsights)
        
        # Verify insights structure
        assert hasattr(insights, 'primary_abnormalities')
        assert hasattr(insights, 'gait_parameters')
        assert hasattr(insights, 'asymmetry_metrics')
        assert hasattr(insights, 'recommendations')
        assert hasattr(insights, 'risk_factors')
        
        # Verify content
        assert 'limping' in insights.primary_abnormalities
        assert len(insights.recommendations) > 0
        assert len(insights.risk_factors) > 0
    
    def test_comprehensive_analysis(self, analysis_engine, sample_classification_results, sample_tracked_pose):
        """Test comprehensive analysis workflow."""
        start_time = time.time()
        
        report = analysis_engine.analyze_comprehensive(
            sample_classification_results, sample_tracked_pose
        )
        
        # Verify return type
        assert isinstance(report, ComprehensiveReport)
        
        # Verify report structure
        assert hasattr(report, 'timestamp')
        assert hasattr(report, 'classification_results')
        assert hasattr(report, 'gait_parameters')
        assert hasattr(report, 'asymmetry_metrics')
        assert hasattr(report, 'temporal_analysis')
        assert hasattr(report, 'clinical_insights')
        assert hasattr(report, 'confidence_score')
        assert hasattr(report, 'processing_time_ms')
        
        # Verify timing
        assert report.timestamp >= start_time
        assert report.processing_time_ms > 0
        
        # Verify confidence score is valid
        assert 0 <= report.confidence_score <= 1
        
        # Verify analysis history is updated
        assert len(analysis_engine.analysis_history) == 1
        assert analysis_engine.analysis_history[0] == report
    
    def test_comprehensive_analysis_multiple_abnormalities(self, analysis_engine, sample_tracked_pose):
        """Test comprehensive analysis with multiple abnormalities."""
        # Create multiple abnormality results
        multiple_results = [
            ClassificationResult(
                abnormality_type='limping',
                confidence=0.75,
                severity_score=0.6,
                affected_limbs=['left_leg']
            ),
            ClassificationResult(
                abnormality_type='balance_issues',
                confidence=0.65,
                severity_score=0.5,
                affected_limbs=['trunk']
            )
        ]
        
        report = analysis_engine.analyze_comprehensive(multiple_results, sample_tracked_pose)
        
        # Verify correlation analysis is performed
        assert report.correlation_analysis is not None
        assert hasattr(report.correlation_analysis, 'primary_abnormality')
        assert hasattr(report.correlation_analysis, 'secondary_abnormalities')
        assert hasattr(report.correlation_analysis, 'correlation_strength')
    
    def test_export_report(self, analysis_engine, sample_classification_results, sample_tracked_pose, tmp_path):
        """Test report export functionality."""
        # Generate a report
        report = analysis_engine.analyze_comprehensive(
            sample_classification_results, sample_tracked_pose
        )
        
        # Export report
        output_path = tmp_path / "test_report.json"
        analysis_engine.export_report(report, str(output_path))
        
        # Verify file was created
        assert output_path.exists()
        
        # Verify file content
        import json
        with open(output_path, 'r') as f:
            exported_data = json.load(f)
        
        assert 'timestamp' in exported_data
        assert 'classification_results' in exported_data
        assert 'gait_parameters' in exported_data
        assert 'metadata' in exported_data
    
    def test_get_analysis_summary(self, analysis_engine, sample_classification_results, sample_tracked_pose):
        """Test analysis summary generation."""
        report = analysis_engine.analyze_comprehensive(
            sample_classification_results, sample_tracked_pose
        )
        
        summary = analysis_engine.get_analysis_summary(report)
        
        # Verify summary structure
        assert 'timestamp' in summary
        assert 'abnormalities_detected' in summary
        assert 'confidence_score' in summary
        assert 'primary_recommendations' in summary
        assert 'key_parameters' in summary
        assert 'asymmetry_severity' in summary
        assert 'processing_time_ms' in summary
        
        # Verify content
        assert 'limping' in summary['abnormalities_detected']
        assert len(summary['primary_recommendations']) <= 3
        assert summary['asymmetry_severity'] in ['minimal', 'mild', 'moderate', 'severe']
    
    def test_parameter_validity_assessment(self, analysis_engine):
        """Test parameter validity assessment."""
        # Test normal parameters
        normal_params = GaitParameters(
            stride_length=1.4,
            cadence=115,
            step_width=0.10,
            swing_time=0.40,
            stance_time=0.60,
            double_support_time=0.15
        )
        
        validity = analysis_engine._assess_parameter_validity(normal_params)
        assert validity > 0.8  # Should be high for normal parameters
        
        # Test abnormal parameters
        abnormal_params = GaitParameters(
            stride_length=0.5,  # Very short
            cadence=50,         # Very slow
            step_width=0.30,    # Very wide
            swing_time=0.20,    # Very short
            stance_time=0.80,   # Very long
            double_support_time=0.40  # Very long
        )
        
        validity = analysis_engine._assess_parameter_validity(abnormal_params)
        assert validity < 0.5  # Should be low for abnormal parameters
    
    def test_asymmetry_severity_classification(self, analysis_engine):
        """Test asymmetry severity classification."""
        # Test minimal asymmetry
        minimal_asymmetry = AsymmetryMetrics(
            left_right_stride_ratio=0.02,
            left_right_swing_ratio=0.03,
            left_right_stance_ratio=0.02,
            temporal_asymmetry=0.03
        )
        
        severity = analysis_engine._classify_asymmetry_severity(minimal_asymmetry)
        assert severity == 'minimal'
        
        # Test severe asymmetry
        severe_asymmetry = AsymmetryMetrics(
            left_right_stride_ratio=0.30,
            left_right_swing_ratio=0.28,
            left_right_stance_ratio=0.25,
            temporal_asymmetry=0.32
        )
        
        severity = analysis_engine._classify_asymmetry_severity(severe_asymmetry)
        assert severity == 'severe'
    
    def test_step_detection(self, analysis_engine):
        """Test step detection algorithm."""
        # Create synthetic ankle height data with clear steps
        ankle_heights = []
        for i in range(100):
            # Create periodic pattern simulating steps
            height = 100 + 20 * np.sin(i * 0.2) + np.random.normal(0, 2)
            ankle_heights.append(height)
        
        ankle_heights = np.array(ankle_heights)
        steps = analysis_engine._detect_steps(ankle_heights)
        
        # Should detect some steps
        assert len(steps) > 0
        assert all(isinstance(step, int) for step in steps)
        assert all(0 < step < len(ankle_heights) - 1 for step in steps)
    
    def test_gait_cycle_detection(self, analysis_engine, sample_pose_sequence):
        """Test gait cycle detection."""
        cycles = analysis_engine._detect_gait_cycles(sample_pose_sequence)
        
        # Should detect at least one cycle
        assert len(cycles) > 0
        
        # Verify cycle structure
        for cycle in cycles:
            assert 'start_frame' in cycle
            assert 'end_frame' in cycle
            assert 'keypoints' in cycle
            assert cycle['start_frame'] < cycle['end_frame']
            assert len(cycle['keypoints']) > 0


class TestClinicalInsightsGenerator:
    """Test suite for ClinicalInsightsGenerator class."""
    
    @pytest.fixture
    def insights_generator(self):
        """Create ClinicalInsightsGenerator instance for testing."""
        return ClinicalInsightsGenerator()
    
    @pytest.fixture
    def sample_patient_context(self):
        """Create sample patient context for testing."""
        return {
            'age': 75,
            'previous_falls': 1,
            'balance_confidence': 0.6,
            'medications': 5,
            'medical_history': ['hypertension', 'diabetes']
        }
    
    def test_initialization(self, insights_generator):
        """Test ClinicalInsightsGenerator initialization."""
        assert insights_generator.clinical_database is not None
        assert insights_generator.risk_models is not None
        assert insights_generator.intervention_protocols is not None
        
        # Verify database structure
        assert 'gait_abnormalities' in insights_generator.clinical_database
        assert 'risk_factors' in insights_generator.clinical_database
    
    def test_generate_advanced_insights(self, insights_generator, sample_patient_context):
        """Test advanced insights generation."""
        # Create test data
        classification_results = [
            ClassificationResult(
                abnormality_type='limping',
                confidence=0.85,
                severity_score=0.6,
                affected_limbs=['left_leg']
            )
        ]
        
        gait_parameters = GaitParameters(
            stride_length=1.0,
            cadence=95,
            step_width=0.15,
            swing_time=0.35,
            stance_time=0.65,
            double_support_time=0.18
        )
        
        asymmetry_metrics = AsymmetryMetrics(
            left_right_stride_ratio=0.12,
            left_right_swing_ratio=0.10,
            left_right_stance_ratio=0.08,
            temporal_asymmetry=0.11
        )
        
        insights = insights_generator.generate_advanced_insights(
            classification_results, gait_parameters, asymmetry_metrics, sample_patient_context
        )
        
        # Verify return type and structure
        from gait_analysis.analysis.clinical_insights import AdvancedClinicalInsights
        assert isinstance(insights, AdvancedClinicalInsights)
        
        # Verify all components are present
        assert insights.clinical_assessment is not None
        assert insights.risk_factors is not None
        assert insights.intervention_recommendations is not None
        assert insights.monitoring_plan is not None
        assert insights.follow_up_schedule is not None
        assert insights.red_flags is not None
        assert insights.patient_education_points is not None
    
    def test_fall_risk_calculation(self, insights_generator, sample_patient_context):
        """Test fall risk calculation."""
        gait_parameters = GaitParameters(
            stride_length=0.8,  # Reduced
            cadence=85,         # Slow
            step_width=0.18,    # Wide
            swing_time=0.30,
            stance_time=0.70,
            double_support_time=0.20
        )
        
        asymmetry_metrics = AsymmetryMetrics(
            left_right_stride_ratio=0.15,
            left_right_swing_ratio=0.12,
            left_right_stance_ratio=0.10,
            temporal_asymmetry=0.14
        )
        
        fall_risk = insights_generator._calculate_fall_risk(
            gait_parameters, asymmetry_metrics, sample_patient_context
        )
        
        # Verify risk calculation structure
        assert 'score' in fall_risk
        assert 'risk_level' in fall_risk
        assert 'confidence' in fall_risk
        
        # Verify reasonable values
        assert 0 <= fall_risk['score'] <= 1
        assert fall_risk['confidence'] > 0
        
        # With poor gait parameters and patient risk factors, should be elevated risk
        from gait_analysis.analysis.clinical_insights import RiskLevel
        assert fall_risk['risk_level'] in [RiskLevel.MODERATE, RiskLevel.HIGH]
    
    def test_severity_assessment(self, insights_generator):
        """Test severity assessment functionality."""
        gait_parameters = GaitParameters(
            stride_length=0.6,  # Very short
            cadence=70,         # Very slow
            step_width=0.25,    # Very wide
            swing_time=0.25,
            stance_time=0.75,
            double_support_time=0.25
        )
        
        asymmetry_metrics = AsymmetryMetrics(
            left_right_stride_ratio=0.30,  # High asymmetry
            left_right_swing_ratio=0.25,
            left_right_stance_ratio=0.20,
            temporal_asymmetry=0.28
        )
        
        severity = insights_generator._assess_severity(
            'limping', gait_parameters, asymmetry_metrics, 0.9
        )
        
        # With poor parameters, should be moderate to severe
        assert severity in [SeverityLevel.MODERATE, SeverityLevel.SEVERE]
    
    def test_intervention_recommendations(self, insights_generator):
        """Test intervention recommendation generation."""
        from gait_analysis.analysis.clinical_insights import ClinicalAssessment
        
        clinical_assessment = ClinicalAssessment(
            primary_diagnosis='limping',
            differential_diagnoses=['hip osteoarthritis', 'knee meniscal tear'],
            severity_assessment=SeverityLevel.MODERATE,
            functional_impact='Moderate functional impact',
            prognosis='Fair prognosis with treatment',
            urgency_level='Semi-urgent - within 2-4 weeks'
        )
        
        recommendations = insights_generator._generate_intervention_recommendations(
            clinical_assessment, [], None
        )
        
        # Should generate relevant recommendations
        assert len(recommendations) > 0
        
        # Verify recommendation structure
        for rec in recommendations:
            assert hasattr(rec, 'intervention_type')
            assert hasattr(rec, 'priority')
            assert hasattr(rec, 'description')
            assert hasattr(rec, 'expected_outcomes')
            assert hasattr(rec, 'contraindications')
            assert hasattr(rec, 'monitoring_parameters')
            assert hasattr(rec, 'estimated_duration')
        
        # Should be sorted by priority
        priorities = [rec.priority for rec in recommendations]
        assert priorities == sorted(priorities)
    
    def test_red_flags_identification(self, insights_generator):
        """Test red flags identification."""
        # Create concerning classification results
        classification_results = [
            ClassificationResult(
                abnormality_type='shuffling',
                confidence=0.9,
                severity_score=0.8,
                affected_limbs=['left_leg', 'right_leg']
            ),
            ClassificationResult(
                abnormality_type='balance_issues',
                confidence=0.85,
                severity_score=0.7,
                affected_limbs=['trunk']
            )
        ]
        
        # Create concerning gait parameters
        gait_parameters = GaitParameters(
            stride_length=0.4,
            cadence=45,  # Extremely slow
            step_width=0.30,
            swing_time=0.20,
            stance_time=0.80,
            double_support_time=0.30
        )
        
        red_flags = insights_generator._identify_red_flags(
            classification_results, gait_parameters, None
        )
        
        # Should identify red flags for multiple severe abnormalities and slow cadence
        assert len(red_flags) > 0
        assert any('multiple' in flag.lower() for flag in red_flags)
        assert any('cadence' in flag.lower() for flag in red_flags)
    
    def test_patient_education_generation(self, insights_generator):
        """Test patient education point generation."""
        from gait_analysis.analysis.clinical_insights import ClinicalAssessment, RiskFactor
        
        clinical_assessment = ClinicalAssessment(
            primary_diagnosis='balance_issues',
            differential_diagnoses=[],
            severity_assessment=SeverityLevel.MODERATE,
            functional_impact='Moderate impact',
            prognosis='Good with intervention',
            urgency_level='Semi-urgent'
        )
        
        risk_factors = [
            RiskFactor(
                factor_name='Fall Risk',
                severity=SeverityLevel.MODERATE,
                confidence=0.8,
                description='Elevated fall risk',
                mitigation_strategies=[]
            )
        ]
        
        education_points = insights_generator._generate_patient_education(
            clinical_assessment, risk_factors
        )
        
        # Should generate relevant education points
        assert len(education_points) > 0
        
        # Should include general safety and specific recommendations
        education_text = ' '.join(education_points).lower()
        assert any(word in education_text for word in ['shoes', 'lighting', 'handrails'])
        assert any(word in education_text for word in ['balance', 'exercise', 'fall'])


if __name__ == '__main__':
    pytest.main([__file__])