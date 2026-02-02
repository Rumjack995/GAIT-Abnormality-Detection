"""Test basic project setup and imports."""

import pytest
import sys
from pathlib import Path

def test_project_structure():
    """Test that all required directories exist."""
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        "gait_analysis",
        "gait_analysis/video_processing",
        "gait_analysis/pose_estimation", 
        "gait_analysis/feature_extraction",
        "gait_analysis/models",
        "gait_analysis/analysis",
        "gait_analysis/utils",
        "tests",
        "data/raw",
        "data/processed",
        "models",
        "notebooks"
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        assert full_path.exists(), f"Required directory missing: {dir_path}"

def test_core_imports():
    """Test that core dependencies can be imported."""
    try:
        import tensorflow as tf
        import cv2
        import mediapipe as mp
        import numpy as np
        import pandas as pd
        import pytest
        import hypothesis
        
        # Verify versions meet minimum requirements (using proper version comparison)
        from packaging import version
        assert version.parse(tf.__version__) >= version.parse("2.13.0"), f"TensorFlow version too old: {tf.__version__}"
        assert version.parse(cv2.__version__) >= version.parse("4.8.0"), f"OpenCV version too old: {cv2.__version__}"
        assert version.parse(mp.__version__) >= version.parse("0.10.0"), f"MediaPipe version too old: {mp.__version__}"
        
    except ImportError as e:
        pytest.fail(f"Failed to import required dependency: {e}")

def test_package_imports():
    """Test that our package modules can be imported."""
    try:
        import gait_analysis
        from gait_analysis.utils.config import Config, get_config
        from gait_analysis.utils.data_structures import ValidationResult, PoseKeypoint
        
        # Test config loading
        config = get_config()
        assert config is not None
        
        # Test basic config access
        video_config = config.video_config
        assert isinstance(video_config, dict)
        
    except ImportError as e:
        pytest.fail(f"Failed to import package module: {e}")

def test_configuration_loading():
    """Test configuration file loading."""
    from gait_analysis.utils.config import Config
    
    # Test default config loading
    config = Config()
    
    # Test basic configuration values
    assert config.get('video.supported_formats') is not None
    assert config.get('pose.confidence_threshold') is not None
    assert config.get('training.batch_size') is not None
    
    # Test non-existent key returns default
    assert config.get('nonexistent.key', 'default') == 'default'

def test_data_structures():
    """Test basic data structure creation."""
    from gait_analysis.utils.data_structures import (
        ValidationResult, PoseKeypoint, GaitParameters
    )
    
    # Test ValidationResult
    result = ValidationResult(
        is_valid=True,
        resolution=(640, 480),
        duration=10.0,
        format="mp4"
    )
    assert result.is_valid is True
    assert result.resolution == (640, 480)
    
    # Test PoseKeypoint
    keypoint = PoseKeypoint(x=0.5, y=0.3, z=0.1, confidence=0.9)
    assert keypoint.confidence == 0.9
    
    # Test GaitParameters
    params = GaitParameters(
        stride_length=1.2,
        cadence=120.0,
        step_width=0.15,
        swing_time=0.4,
        stance_time=0.6,
        double_support_time=0.2
    )
    assert params.stride_length == 1.2