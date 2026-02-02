"""
Clinical Insights Generation System for Gait Abnormality Analysis.

This module implements advanced clinical recommendation systems, risk factor
identification algorithms, and intervention strategy suggestions based on
detected gait patterns and abnormalities.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from ..utils.data_structures import (
    ClassificationResult,
    GaitParameters,
    AsymmetryMetrics,
    ClinicalInsights
)


class SeverityLevel(Enum):
    """Severity levels for clinical conditions."""
    MINIMAL = "minimal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class RiskLevel(Enum):
    """Risk levels for clinical outcomes."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class RiskFactor:
    """Individual risk factor with severity and confidence."""
    factor_name: str
    severity: SeverityLevel
    confidence: float
    description: str
    mitigation_strategies: List[str]


@dataclass
class InterventionRecommendation:
    """Clinical intervention recommendation."""
    intervention_type: str
    priority: int  # 1 = highest priority
    description: str
    expected_outcomes: List[str]
    contraindications: List[str]
    monitoring_parameters: List[str]
    estimated_duration: str


@dataclass
class ClinicalAssessment:
    """Comprehensive clinical assessment."""
    primary_diagnosis: str
    differential_diagnoses: List[str]
    severity_assessment: SeverityLevel
    functional_impact: str
    prognosis: str
    urgency_level: str


@dataclass
class AdvancedClinicalInsights:
    """Advanced clinical insights with detailed recommendations."""
    clinical_assessment: ClinicalAssessment
    risk_factors: List[RiskFactor]
    intervention_recommendations: List[InterventionRecommendation]
    monitoring_plan: Dict[str, Any]
    follow_up_schedule: Dict[str, str]
    red_flags: List[str]
    patient_education_points: List[str]


class ClinicalInsightsGenerator:
    """
    Advanced clinical insights generation system.
    
    This class provides sophisticated clinical analysis including:
    - Risk factor identification and stratification
    - Evidence-based intervention recommendations
    - Clinical assessment and differential diagnosis
    - Monitoring and follow-up planning
    - Patient education recommendations
    """
    
    def __init__(self):
        """Initialize the clinical insights generator."""
        self.clinical_database = self._load_clinical_database()
        self.risk_models = self._initialize_risk_models()
        self.intervention_protocols = self._load_intervention_protocols()
    
    def _load_clinical_database(self) -> Dict[str, Any]:
        """Load comprehensive clinical knowledge database."""
        return {
            'gait_abnormalities': {
                'limping': {
                    'clinical_name': 'Antalgic Gait',
                    'icd_codes': ['M25.50', 'R26.0'],
                    'common_causes': [
                        'Lower extremity pain',
                        'Joint pathology (hip, knee, ankle)',
                        'Muscle strain or weakness',
                        'Fracture or injury',
                        'Arthritis'
                    ],
                    'differential_diagnoses': [
                        'Hip osteoarthritis',
                        'Knee meniscal tear',
                        'Ankle sprain',
                        'Plantar fasciitis',
                        'Stress fracture'
                    ],
                    'severity_indicators': {
                        'mild': {'stride_asymmetry': 0.05, 'pain_scale': 3},
                        'moderate': {'stride_asymmetry': 0.15, 'pain_scale': 6},
                        'severe': {'stride_asymmetry': 0.25, 'pain_scale': 8}
                    }
                },
                'shuffling': {
                    'clinical_name': 'Shuffling Gait',
                    'icd_codes': ['G20', 'R26.1'],
                    'common_causes': [
                        'Parkinson\'s disease',
                        'Normal pressure hydrocephalus',
                        'Medication-induced parkinsonism',
                        'Progressive supranuclear palsy',
                        'Multiple system atrophy'
                    ],
                    'differential_diagnoses': [
                        'Parkinson\'s disease',
                        'Drug-induced parkinsonism',
                        'Normal pressure hydrocephalus',
                        'Vascular parkinsonism',
                        'Dementia with Lewy bodies'
                    ],
                    'severity_indicators': {
                        'mild': {'step_length_reduction': 0.1, 'freezing_episodes': 0},
                        'moderate': {'step_length_reduction': 0.3, 'freezing_episodes': 2},
                        'severe': {'step_length_reduction': 0.5, 'freezing_episodes': 5}
                    }
                },
                'irregular_stride': {
                    'clinical_name': 'Ataxic Gait',
                    'icd_codes': ['R26.0', 'G11.9'],
                    'common_causes': [
                        'Cerebellar dysfunction',
                        'Vestibular disorders',
                        'Sensory ataxia',
                        'Alcohol intoxication',
                        'Medication side effects'
                    ],
                    'differential_diagnoses': [
                        'Cerebellar ataxia',
                        'Vestibular dysfunction',
                        'Peripheral neuropathy',
                        'Vitamin B12 deficiency',
                        'Multiple sclerosis'
                    ],
                    'severity_indicators': {
                        'mild': {'gait_variability': 0.1, 'balance_confidence': 0.8},
                        'moderate': {'gait_variability': 0.2, 'balance_confidence': 0.6},
                        'severe': {'gait_variability': 0.4, 'balance_confidence': 0.3}
                    }
                },
                'balance_issues': {
                    'clinical_name': 'Balance Impairment',
                    'icd_codes': ['R26.81', 'H81.9'],
                    'common_causes': [
                        'Vestibular disorders',
                        'Cerebellar pathology',
                        'Peripheral neuropathy',
                        'Visual impairment',
                        'Medication effects'
                    ],
                    'differential_diagnoses': [
                        'Benign paroxysmal positional vertigo',
                        'Vestibular neuritis',
                        'Meniere\'s disease',
                        'Cerebellar stroke',
                        'Diabetic neuropathy'
                    ],
                    'severity_indicators': {
                        'mild': {'fall_risk_score': 2, 'tug_time': 12},
                        'moderate': {'fall_risk_score': 4, 'tug_time': 15},
                        'severe': {'fall_risk_score': 7, 'tug_time': 20}
                    }
                }
            },
            'risk_factors': {
                'fall_risk': {
                    'high_risk_indicators': [
                        'Previous falls',
                        'Balance impairment',
                        'Gait abnormalities',
                        'Medication use (>4 medications)',
                        'Cognitive impairment'
                    ],
                    'protective_factors': [
                        'Regular exercise',
                        'Good vision',
                        'Safe home environment',
                        'Appropriate footwear',
                        'Social support'
                    ]
                },
                'functional_decline': {
                    'risk_indicators': [
                        'Reduced walking speed',
                        'Decreased stride length',
                        'Increased gait variability',
                        'Balance confidence decline',
                        'Activity avoidance'
                    ]
                }
            }
        }
    
    def _initialize_risk_models(self) -> Dict[str, Any]:
        """Initialize risk prediction models."""
        return {
            'fall_risk_model': {
                'weights': {
                    'gait_speed': -2.5,
                    'stride_variability': 1.8,
                    'balance_confidence': -1.2,
                    'previous_falls': 2.0,
                    'age': 0.05
                },
                'thresholds': {
                    'low': 0.2,
                    'moderate': 0.5,
                    'high': 0.8
                }
            },
            'functional_decline_model': {
                'weights': {
                    'gait_asymmetry': 1.5,
                    'cadence_reduction': 1.2,
                    'step_width_increase': 0.8,
                    'temporal_variability': 1.0
                },
                'thresholds': {
                    'low': 0.3,
                    'moderate': 0.6,
                    'high': 0.8
                }
            }
        }
    
    def _load_intervention_protocols(self) -> Dict[str, Any]:
        """Load evidence-based intervention protocols."""
        return {
            'physical_therapy': {
                'gait_training': {
                    'description': 'Structured gait retraining program',
                    'duration': '6-12 weeks',
                    'frequency': '2-3 sessions per week',
                    'outcomes': [
                        'Improved gait symmetry',
                        'Increased walking speed',
                        'Enhanced balance confidence'
                    ],
                    'contraindications': [
                        'Acute fracture',
                        'Severe pain',
                        'Unstable medical condition'
                    ]
                },
                'balance_training': {
                    'description': 'Progressive balance and coordination exercises',
                    'duration': '8-16 weeks',
                    'frequency': '3 sessions per week',
                    'outcomes': [
                        'Reduced fall risk',
                        'Improved postural control',
                        'Enhanced functional mobility'
                    ]
                }
            },
            'medical_management': {
                'neurological_evaluation': {
                    'description': 'Comprehensive neurological assessment',
                    'urgency': 'within 2-4 weeks',
                    'indications': [
                        'Progressive gait deterioration',
                        'Neurological signs',
                        'Cognitive changes'
                    ]
                },
                'medication_review': {
                    'description': 'Systematic medication evaluation',
                    'urgency': 'within 1-2 weeks',
                    'focus_areas': [
                        'Sedating medications',
                        'Anticholinergics',
                        'Antipsychotics',
                        'Polypharmacy'
                    ]
                }
            },
            'assistive_devices': {
                'walking_aids': {
                    'cane': {
                        'indications': ['Mild balance issues', 'Unilateral weakness'],
                        'contraindications': ['Severe bilateral weakness', 'Cognitive impairment']
                    },
                    'walker': {
                        'indications': ['Moderate to severe balance issues', 'Bilateral weakness'],
                        'contraindications': ['Severe cognitive impairment', 'Upper extremity weakness']
                    }
                }
            }
        }
    
    def generate_advanced_insights(self,
                                 classification_results: List[ClassificationResult],
                                 gait_parameters: GaitParameters,
                                 asymmetry_metrics: AsymmetryMetrics,
                                 patient_context: Optional[Dict[str, Any]] = None) -> AdvancedClinicalInsights:
        """
        Generate comprehensive clinical insights with advanced recommendations.
        
        Args:
            classification_results: Gait classification results
            gait_parameters: Calculated gait parameters
            asymmetry_metrics: Asymmetry analysis results
            patient_context: Optional patient information (age, medical history, etc.)
            
        Returns:
            Advanced clinical insights with detailed recommendations
        """
        # Perform clinical assessment
        clinical_assessment = self._perform_clinical_assessment(
            classification_results, gait_parameters, asymmetry_metrics, patient_context
        )
        
        # Identify and stratify risk factors
        risk_factors = self._identify_risk_factors(
            classification_results, gait_parameters, asymmetry_metrics, patient_context
        )
        
        # Generate intervention recommendations
        intervention_recommendations = self._generate_intervention_recommendations(
            clinical_assessment, risk_factors, patient_context
        )
        
        # Create monitoring plan
        monitoring_plan = self._create_monitoring_plan(
            clinical_assessment, intervention_recommendations
        )
        
        # Develop follow-up schedule
        follow_up_schedule = self._create_follow_up_schedule(
            clinical_assessment, risk_factors
        )
        
        # Identify red flags
        red_flags = self._identify_red_flags(
            classification_results, gait_parameters, patient_context
        )
        
        # Generate patient education points
        patient_education_points = self._generate_patient_education(
            clinical_assessment, risk_factors
        )
        
        return AdvancedClinicalInsights(
            clinical_assessment=clinical_assessment,
            risk_factors=risk_factors,
            intervention_recommendations=intervention_recommendations,
            monitoring_plan=monitoring_plan,
            follow_up_schedule=follow_up_schedule,
            red_flags=red_flags,
            patient_education_points=patient_education_points
        )
    
    def _perform_clinical_assessment(self,
                                   classification_results: List[ClassificationResult],
                                   gait_parameters: GaitParameters,
                                   asymmetry_metrics: AsymmetryMetrics,
                                   patient_context: Optional[Dict[str, Any]]) -> ClinicalAssessment:
        """Perform comprehensive clinical assessment."""
        # Identify primary abnormality
        primary_result = max(classification_results, key=lambda x: x.confidence)
        primary_diagnosis = primary_result.abnormality_type
        
        # Get differential diagnoses
        if primary_diagnosis in self.clinical_database['gait_abnormalities']:
            differential_diagnoses = self.clinical_database['gait_abnormalities'][primary_diagnosis]['differential_diagnoses']
        else:
            differential_diagnoses = []
        
        # Assess severity
        severity_assessment = self._assess_severity(
            primary_diagnosis, gait_parameters, asymmetry_metrics, primary_result.confidence
        )
        
        # Determine functional impact
        functional_impact = self._assess_functional_impact(
            gait_parameters, asymmetry_metrics, severity_assessment
        )
        
        # Determine prognosis
        prognosis = self._determine_prognosis(
            primary_diagnosis, severity_assessment, patient_context
        )
        
        # Assess urgency
        urgency_level = self._assess_urgency(
            primary_diagnosis, severity_assessment, gait_parameters
        )
        
        return ClinicalAssessment(
            primary_diagnosis=primary_diagnosis,
            differential_diagnoses=differential_diagnoses,
            severity_assessment=severity_assessment,
            functional_impact=functional_impact,
            prognosis=prognosis,
            urgency_level=urgency_level
        )
    
    def _identify_risk_factors(self,
                             classification_results: List[ClassificationResult],
                             gait_parameters: GaitParameters,
                             asymmetry_metrics: AsymmetryMetrics,
                             patient_context: Optional[Dict[str, Any]]) -> List[RiskFactor]:
        """Identify and stratify risk factors."""
        risk_factors = []
        
        # Fall risk assessment
        fall_risk = self._calculate_fall_risk(
            gait_parameters, asymmetry_metrics, patient_context
        )
        
        if fall_risk['risk_level'] != RiskLevel.LOW:
            risk_factors.append(RiskFactor(
                factor_name='Fall Risk',
                severity=self._risk_level_to_severity(fall_risk['risk_level']),
                confidence=fall_risk['confidence'],
                description=f"Elevated fall risk based on gait analysis (score: {fall_risk['score']:.2f})",
                mitigation_strategies=[
                    'Balance training program',
                    'Home safety assessment',
                    'Medication review',
                    'Vision screening'
                ]
            ))
        
        # Functional decline risk
        functional_risk = self._calculate_functional_decline_risk(
            gait_parameters, asymmetry_metrics
        )
        
        if functional_risk['risk_level'] != RiskLevel.LOW:
            risk_factors.append(RiskFactor(
                factor_name='Functional Decline',
                severity=self._risk_level_to_severity(functional_risk['risk_level']),
                confidence=functional_risk['confidence'],
                description=f"Risk of functional decline (score: {functional_risk['score']:.2f})",
                mitigation_strategies=[
                    'Regular exercise program',
                    'Gait training',
                    'Strength training',
                    'Activity monitoring'
                ]
            ))
        
        # Mobility limitation risk
        if gait_parameters.stride_length < 1.0 or gait_parameters.cadence < 90:
            risk_factors.append(RiskFactor(
                factor_name='Mobility Limitation',
                severity=SeverityLevel.MODERATE,
                confidence=0.8,
                description="Reduced gait speed and stride length indicating mobility limitations",
                mitigation_strategies=[
                    'Physical therapy evaluation',
                    'Assistive device assessment',
                    'Endurance training',
                    'Pain management'
                ]
            ))
        
        return risk_factors
    
    def _generate_intervention_recommendations(self,
                                            clinical_assessment: ClinicalAssessment,
                                            risk_factors: List[RiskFactor],
                                            patient_context: Optional[Dict[str, Any]]) -> List[InterventionRecommendation]:
        """Generate evidence-based intervention recommendations."""
        recommendations = []
        
        # Primary diagnosis-based recommendations
        primary_diagnosis = clinical_assessment.primary_diagnosis
        
        if primary_diagnosis == 'limping':
            recommendations.extend([
                InterventionRecommendation(
                    intervention_type='Physical Therapy',
                    priority=1,
                    description='Gait training and strengthening program for affected limb',
                    expected_outcomes=['Improved gait symmetry', 'Reduced pain', 'Enhanced function'],
                    contraindications=['Acute fracture', 'Severe pain'],
                    monitoring_parameters=['Gait symmetry', 'Pain levels', 'Functional scores'],
                    estimated_duration='6-12 weeks'
                ),
                InterventionRecommendation(
                    intervention_type='Medical Evaluation',
                    priority=2,
                    description='Orthopedic or rheumatology consultation for underlying pathology',
                    expected_outcomes=['Accurate diagnosis', 'Targeted treatment'],
                    contraindications=[],
                    monitoring_parameters=['Diagnostic results', 'Treatment response'],
                    estimated_duration='2-4 weeks'
                )
            ])
        
        elif primary_diagnosis == 'shuffling':
            recommendations.extend([
                InterventionRecommendation(
                    intervention_type='Neurological Evaluation',
                    priority=1,
                    description='Comprehensive neurological assessment for movement disorders',
                    expected_outcomes=['Accurate diagnosis', 'Appropriate treatment'],
                    contraindications=[],
                    monitoring_parameters=['Neurological examination', 'Imaging results'],
                    estimated_duration='2-4 weeks'
                ),
                InterventionRecommendation(
                    intervention_type='Gait Training with Cueing',
                    priority=2,
                    description='Specialized gait training with auditory/visual cues',
                    expected_outcomes=['Improved step length', 'Reduced freezing'],
                    contraindications=['Severe cognitive impairment'],
                    monitoring_parameters=['Step length', 'Freezing episodes', 'Walking speed'],
                    estimated_duration='8-16 weeks'
                )
            ])
        
        elif primary_diagnosis == 'irregular_stride':
            recommendations.extend([
                InterventionRecommendation(
                    intervention_type='Balance Training',
                    priority=1,
                    description='Progressive balance and coordination exercises',
                    expected_outcomes=['Improved balance', 'Reduced gait variability'],
                    contraindications=['Acute vestibular symptoms'],
                    monitoring_parameters=['Balance confidence', 'Gait variability', 'Fall incidents'],
                    estimated_duration='8-12 weeks'
                ),
                InterventionRecommendation(
                    intervention_type='Vestibular Assessment',
                    priority=2,
                    description='Comprehensive vestibular function evaluation',
                    expected_outcomes=['Identify vestibular dysfunction', 'Targeted treatment'],
                    contraindications=[],
                    monitoring_parameters=['Vestibular test results', 'Symptom improvement'],
                    estimated_duration='1-2 weeks'
                )
            ])
        
        elif primary_diagnosis == 'balance_issues':
            recommendations.extend([
                InterventionRecommendation(
                    intervention_type='Fall Prevention Program',
                    priority=1,
                    description='Comprehensive fall prevention with balance training',
                    expected_outcomes=['Reduced fall risk', 'Improved confidence'],
                    contraindications=['Unstable medical condition'],
                    monitoring_parameters=['Fall incidents', 'Balance scores', 'Activity levels'],
                    estimated_duration='12-16 weeks'
                ),
                InterventionRecommendation(
                    intervention_type='Home Safety Assessment',
                    priority=2,
                    description='Occupational therapy home safety evaluation',
                    expected_outcomes=['Safer home environment', 'Reduced fall risk'],
                    contraindications=[],
                    monitoring_parameters=['Home modifications', 'Safety compliance'],
                    estimated_duration='1-2 weeks'
                )
            ])
        
        # Risk factor-based recommendations
        for risk_factor in risk_factors:
            if risk_factor.factor_name == 'Fall Risk' and risk_factor.severity in [SeverityLevel.MODERATE, SeverityLevel.SEVERE]:
                recommendations.append(
                    InterventionRecommendation(
                        intervention_type='Assistive Device Assessment',
                        priority=3,
                        description='Evaluation for appropriate walking aids',
                        expected_outcomes=['Enhanced stability', 'Increased confidence'],
                        contraindications=['Cognitive impairment', 'Upper extremity weakness'],
                        monitoring_parameters=['Device compliance', 'Stability improvement'],
                        estimated_duration='1-2 weeks'
                    )
                )
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority)
        
        return recommendations
    
    def _create_monitoring_plan(self,
                              clinical_assessment: ClinicalAssessment,
                              intervention_recommendations: List[InterventionRecommendation]) -> Dict[str, Any]:
        """Create comprehensive monitoring plan."""
        monitoring_plan = {
            'primary_outcomes': [],
            'secondary_outcomes': [],
            'safety_parameters': [],
            'assessment_schedule': {},
            'alert_thresholds': {}
        }
        
        # Primary outcomes based on diagnosis
        if clinical_assessment.primary_diagnosis == 'limping':
            monitoring_plan['primary_outcomes'] = [
                'Gait symmetry index',
                'Pain levels (0-10 scale)',
                'Walking speed'
            ]
        elif clinical_assessment.primary_diagnosis == 'shuffling':
            monitoring_plan['primary_outcomes'] = [
                'Step length',
                'Freezing episodes',
                'UPDRS gait score'
            ]
        elif clinical_assessment.primary_diagnosis in ['irregular_stride', 'balance_issues']:
            monitoring_plan['primary_outcomes'] = [
                'Balance confidence scale',
                'Gait variability',
                'Fall incidents'
            ]
        
        # Secondary outcomes
        monitoring_plan['secondary_outcomes'] = [
            'Quality of life scores',
            'Activity levels',
            'Functional independence'
        ]
        
        # Safety parameters
        monitoring_plan['safety_parameters'] = [
            'Fall incidents',
            'Injury reports',
            'Adverse events'
        ]
        
        # Assessment schedule
        monitoring_plan['assessment_schedule'] = {
            'baseline': 'Initial assessment completed',
            '2_weeks': 'Safety check and early response',
            '6_weeks': 'Mid-intervention assessment',
            '12_weeks': 'Primary outcome evaluation',
            '6_months': 'Long-term follow-up'
        }
        
        # Alert thresholds
        monitoring_plan['alert_thresholds'] = {
            'fall_incident': 'Immediate medical review',
            'pain_increase': '>2 point increase on pain scale',
            'function_decline': '>20% decrease in walking speed'
        }
        
        return monitoring_plan
    
    def _create_follow_up_schedule(self,
                                 clinical_assessment: ClinicalAssessment,
                                 risk_factors: List[RiskFactor]) -> Dict[str, str]:
        """Create appropriate follow-up schedule."""
        follow_up_schedule = {}
        
        # Base schedule on severity and urgency
        if clinical_assessment.severity_assessment == SeverityLevel.SEVERE:
            follow_up_schedule['immediate'] = 'Within 1 week'
            follow_up_schedule['short_term'] = '2-4 weeks'
            follow_up_schedule['medium_term'] = '6-8 weeks'
        elif clinical_assessment.severity_assessment == SeverityLevel.MODERATE:
            follow_up_schedule['short_term'] = '2-3 weeks'
            follow_up_schedule['medium_term'] = '6-8 weeks'
            follow_up_schedule['long_term'] = '3-6 months'
        else:
            follow_up_schedule['medium_term'] = '4-6 weeks'
            follow_up_schedule['long_term'] = '3-6 months'
        
        # Adjust based on risk factors
        high_risk_factors = [rf for rf in risk_factors if rf.severity in [SeverityLevel.MODERATE, SeverityLevel.SEVERE]]
        if high_risk_factors:
            follow_up_schedule['risk_monitoring'] = 'Monthly for 6 months'
        
        return follow_up_schedule
    
    def _identify_red_flags(self,
                          classification_results: List[ClassificationResult],
                          gait_parameters: GaitParameters,
                          patient_context: Optional[Dict[str, Any]]) -> List[str]:
        """Identify clinical red flags requiring immediate attention."""
        red_flags = []
        
        # Severe gait abnormalities
        severe_abnormalities = [r for r in classification_results 
                              if r.abnormality_type != 'normal' and r.confidence > 0.8]
        
        if len(severe_abnormalities) > 1:
            red_flags.append("Multiple severe gait abnormalities detected - consider urgent neurological evaluation")
        
        # Extremely slow gait
        if gait_parameters.cadence < 60:
            red_flags.append("Severely reduced cadence - high fall risk and functional decline concern")
        
        # Severe asymmetry
        if hasattr(gait_parameters, 'asymmetry_ratio') and gait_parameters.asymmetry_ratio > 0.3:
            red_flags.append("Severe gait asymmetry - investigate for acute pathology")
        
        # Progressive symptoms (if patient context available)
        if patient_context and patient_context.get('symptom_progression') == 'rapid':
            red_flags.append("Rapid symptom progression - urgent medical evaluation recommended")
        
        return red_flags
    
    def _generate_patient_education(self,
                                  clinical_assessment: ClinicalAssessment,
                                  risk_factors: List[RiskFactor]) -> List[str]:
        """Generate patient education points."""
        education_points = []
        
        # General gait safety
        education_points.extend([
            "Wear appropriate, well-fitting shoes with good traction",
            "Ensure adequate lighting in walking areas",
            "Remove tripping hazards from walkways",
            "Use handrails when available"
        ])
        
        # Diagnosis-specific education
        if clinical_assessment.primary_diagnosis == 'limping':
            education_points.extend([
                "Avoid activities that worsen pain",
                "Apply ice or heat as recommended",
                "Perform prescribed exercises regularly"
            ])
        elif clinical_assessment.primary_diagnosis == 'shuffling':
            education_points.extend([
                "Practice walking with deliberate, larger steps",
                "Use visual or auditory cues to improve gait",
                "Take medications as prescribed"
            ])
        elif clinical_assessment.primary_diagnosis in ['irregular_stride', 'balance_issues']:
            education_points.extend([
                "Practice balance exercises daily",
                "Avoid sudden movements or direction changes",
                "Consider using assistive devices when recommended"
            ])
        
        # Risk factor-specific education
        fall_risk_factors = [rf for rf in risk_factors if 'fall' in rf.factor_name.lower()]
        if fall_risk_factors:
            education_points.extend([
                "Report any falls or near-falls immediately",
                "Consider wearing a medical alert device",
                "Have regular vision and hearing checks"
            ])
        
        return education_points
    
    def _assess_severity(self,
                       primary_diagnosis: str,
                       gait_parameters: GaitParameters,
                       asymmetry_metrics: AsymmetryMetrics,
                       confidence: float) -> SeverityLevel:
        """Assess severity of the primary diagnosis."""
        if primary_diagnosis not in self.clinical_database['gait_abnormalities']:
            return SeverityLevel.MILD
        
        severity_indicators = self.clinical_database['gait_abnormalities'][primary_diagnosis]['severity_indicators']
        
        # Calculate severity score based on multiple factors
        severity_score = 0
        
        # Confidence factor
        severity_score += confidence * 0.3
        
        # Gait parameter factors
        if gait_parameters.stride_length < 1.0:
            severity_score += 0.2
        if gait_parameters.cadence < 90:
            severity_score += 0.2
        
        # Asymmetry factors
        max_asymmetry = max(
            asymmetry_metrics.left_right_stride_ratio,
            asymmetry_metrics.temporal_asymmetry
        )
        severity_score += min(max_asymmetry * 2, 0.3)
        
        # Map score to severity level
        if severity_score < 0.3:
            return SeverityLevel.MILD
        elif severity_score < 0.6:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.SEVERE
    
    def _assess_functional_impact(self,
                                gait_parameters: GaitParameters,
                                asymmetry_metrics: AsymmetryMetrics,
                                severity: SeverityLevel) -> str:
        """Assess functional impact of gait abnormalities."""
        if severity == SeverityLevel.SEVERE:
            return "Significant functional limitation with reduced independence"
        elif severity == SeverityLevel.MODERATE:
            return "Moderate functional impact with some activity limitations"
        else:
            return "Minimal functional impact with preserved independence"
    
    def _determine_prognosis(self,
                           primary_diagnosis: str,
                           severity: SeverityLevel,
                           patient_context: Optional[Dict[str, Any]]) -> str:
        """Determine prognosis based on diagnosis and severity."""
        prognosis_map = {
            'limping': {
                SeverityLevel.MILD: "Good prognosis with appropriate treatment",
                SeverityLevel.MODERATE: "Fair prognosis, improvement expected with intervention",
                SeverityLevel.SEVERE: "Guarded prognosis, may require ongoing management"
            },
            'shuffling': {
                SeverityLevel.MILD: "Stable with potential for improvement",
                SeverityLevel.MODERATE: "Progressive condition, management can slow decline",
                SeverityLevel.SEVERE: "Progressive condition with significant functional impact"
            },
            'irregular_stride': {
                SeverityLevel.MILD: "Good potential for improvement with training",
                SeverityLevel.MODERATE: "Moderate improvement expected with intervention",
                SeverityLevel.SEVERE: "Limited improvement, focus on safety and function"
            },
            'balance_issues': {
                SeverityLevel.MILD: "Excellent prognosis with balance training",
                SeverityLevel.MODERATE: "Good prognosis with comprehensive intervention",
                SeverityLevel.SEVERE: "Guarded prognosis, high fall risk"
            }
        }
        
        return prognosis_map.get(primary_diagnosis, {}).get(severity, "Prognosis depends on underlying cause")
    
    def _assess_urgency(self,
                      primary_diagnosis: str,
                      severity: SeverityLevel,
                      gait_parameters: GaitParameters) -> str:
        """Assess urgency level for medical attention."""
        if severity == SeverityLevel.SEVERE:
            return "Urgent - within 1-2 weeks"
        elif severity == SeverityLevel.MODERATE:
            return "Semi-urgent - within 2-4 weeks"
        else:
            return "Routine - within 4-8 weeks"
    
    def _calculate_fall_risk(self,
                           gait_parameters: GaitParameters,
                           asymmetry_metrics: AsymmetryMetrics,
                           patient_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate fall risk using validated risk model."""
        model = self.risk_models['fall_risk_model']
        
        # Calculate risk score
        score = 0
        
        # Gait speed factor (assuming normal walking speed ~1.2 m/s)
        estimated_speed = (gait_parameters.stride_length * gait_parameters.cadence) / 60
        score += model['weights']['gait_speed'] * (1.2 - estimated_speed)
        
        # Stride variability (using asymmetry as proxy)
        stride_variability = asymmetry_metrics.temporal_asymmetry
        score += model['weights']['stride_variability'] * stride_variability
        
        # Patient context factors
        if patient_context:
            if 'balance_confidence' in patient_context:
                score += model['weights']['balance_confidence'] * (1 - patient_context['balance_confidence'])
            if 'previous_falls' in patient_context:
                score += model['weights']['previous_falls'] * patient_context['previous_falls']
            if 'age' in patient_context:
                score += model['weights']['age'] * max(0, patient_context['age'] - 65)
        
        # Normalize score to 0-1 range
        normalized_score = 1 / (1 + np.exp(-score))  # Sigmoid function
        
        # Determine risk level
        if normalized_score < model['thresholds']['low']:
            risk_level = RiskLevel.LOW
        elif normalized_score < model['thresholds']['moderate']:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.HIGH
        
        return {
            'score': normalized_score,
            'risk_level': risk_level,
            'confidence': 0.8  # Model confidence
        }
    
    def _calculate_functional_decline_risk(self,
                                         gait_parameters: GaitParameters,
                                         asymmetry_metrics: AsymmetryMetrics) -> Dict[str, Any]:
        """Calculate functional decline risk."""
        model = self.risk_models['functional_decline_model']
        
        score = 0
        
        # Gait asymmetry
        max_asymmetry = max(
            asymmetry_metrics.left_right_stride_ratio,
            asymmetry_metrics.temporal_asymmetry
        )
        score += model['weights']['gait_asymmetry'] * max_asymmetry
        
        # Cadence reduction (assuming normal ~115 steps/min)
        cadence_reduction = max(0, (115 - gait_parameters.cadence) / 115)
        score += model['weights']['cadence_reduction'] * cadence_reduction
        
        # Step width increase (assuming normal ~0.1m)
        step_width_increase = max(0, gait_parameters.step_width - 0.1)
        score += model['weights']['step_width_increase'] * step_width_increase
        
        # Temporal variability (using asymmetry as proxy)
        score += model['weights']['temporal_variability'] * asymmetry_metrics.temporal_asymmetry
        
        # Normalize score
        normalized_score = min(score, 1.0)
        
        # Determine risk level
        if normalized_score < model['thresholds']['low']:
            risk_level = RiskLevel.LOW
        elif normalized_score < model['thresholds']['moderate']:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.HIGH
        
        return {
            'score': normalized_score,
            'risk_level': risk_level,
            'confidence': 0.75
        }
    
    def _risk_level_to_severity(self, risk_level: RiskLevel) -> SeverityLevel:
        """Convert risk level to severity level."""
        mapping = {
            RiskLevel.LOW: SeverityLevel.MINIMAL,
            RiskLevel.MODERATE: SeverityLevel.MODERATE,
            RiskLevel.HIGH: SeverityLevel.SEVERE,
            RiskLevel.VERY_HIGH: SeverityLevel.CRITICAL
        }
        return mapping.get(risk_level, SeverityLevel.MILD)