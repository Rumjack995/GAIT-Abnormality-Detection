"""
Gait Abnormality Detection System

A deep learning system that analyzes video input to detect and classify gait abnormalities,
providing medical professionals and researchers with automated pattern recognition capabilities.
"""

from .classification import GaitClassifier

__version__ = "0.1.0"
__author__ = "Gait Analysis Team"

__all__ = ['GaitClassifier']