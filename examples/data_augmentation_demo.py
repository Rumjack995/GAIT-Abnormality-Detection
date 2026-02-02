#!/usr/bin/env python3
"""
Data Augmentation Demo

This script demonstrates the data augmentation capabilities for gait analysis training data.
It shows how to use the DataAugmentation class to augment pose sequences with various
transformations including rotation, scaling, temporal modifications, and noise.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import logging
from gait_analysis.feature_extraction import DataAugmentation
from gait_analysis.utils.data_structures import (
    PoseKeypoint, PoseSequence, TrainingExample
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_pose_sequence(n_frames=30):
    """Create a sample pose sequence for demonstration."""
    keypoints_sequence = []
    timestamps = []
    confidence_scores = []
    
    for frame_idx in range(n_frames):
        frame_keypoints = []
        
        # Create 33 keypoints (MediaPipe standard)
        for kp_idx in range(33):
            # Simulate walking motion with some variation
            x = 100 + np.sin(frame_idx * 0.2) * 20
            y = 200 + np.cos(frame_idx * 0.2) * 10
            z = 0 + np.random.normal(0, 1)
            confidence = 0.9
            
            frame_keypoints.append(PoseKeypoint(x=x, y=y, z=z, confidence=confidence))
        
        keypoints_sequence.append(frame_keypoints)
        timestamps.append(frame_idx / 30.0)
        confidence_scores.append(0.9)
    
    return PoseSequence(
        keypoints=keypoints_sequence,
        timestamps=timestamps,
        confidence_scores=confidence_scores
    )


def create_sample_training_examples(n_examples=5):
    """Create sample training examples."""
    examples = []
    
    for i in range(n_examples):
        pose_sequence = create_sample_pose_sequence(30)
        
        example = TrainingExample(
            video_path=f"sample_video_{i}.mp4",
            pose_sequence=pose_sequence,
            ground_truth_label="normal" if i % 2 == 0 else "abnormal",
            severity_score=0.0 if i % 2 == 0 else 0.5,
            metadata={"duration": 1.0, "subject_id": f"subject_{i}"}
        )
        examples.append(example)
    
    return examples


def demonstrate_augmentation_pipeline():
    """Demonstrate the complete augmentation pipeline."""
    logger.info("=== Data Augmentation Demo ===")
    
    # Create sample training data
    logger.info("Creating sample training data...")
    original_examples = create_sample_training_examples(5)
    logger.info(f"Created {len(original_examples)} original training examples")
    
    # Initialize data augmentation with custom parameters
    logger.info("Initializing DataAugmentation...")
    data_augmentation = DataAugmentation(
        rotation_range=20.0,
        scale_range=(0.8, 1.2),
        noise_std=0.02,
        temporal_stretch_range=(0.8, 1.2),
        flip_probability=0.5
    )
    
    # Create augmentation pipeline configuration
    logger.info("Creating augmentation pipeline...")
    dataset_size = len(original_examples)
    target_size = 20  # Want to augment to 20 examples
    
    pipeline_config = data_augmentation.create_augmentation_pipeline(
        dataset_size, target_size
    )
    
    logger.info(f"Pipeline configuration:")
    logger.info(f"  - Augmentation factor: {pipeline_config['augmentation_factor']}")
    logger.info(f"  - Strategies: {pipeline_config['strategies']}")
    logger.info(f"  - Target size: {pipeline_config['total_augmented_size']}")
    logger.info(f"  - Parameters: {pipeline_config['parameters']}")
    
    # Apply augmentation
    logger.info("Applying augmentation to training dataset...")
    augmented_examples = data_augmentation.augment_training_dataset(
        original_examples, 
        augmentation_factor=pipeline_config['augmentation_factor']
    )
    
    logger.info(f"Augmentation complete!")
    logger.info(f"  - Original examples: {len(original_examples)}")
    logger.info(f"  - Augmented examples: {len(augmented_examples)}")
    logger.info(f"  - Total increase: {len(augmented_examples) - len(original_examples)}")
    
    # Analyze augmented examples
    logger.info("Analyzing augmented examples...")
    original_paths = [ex.video_path for ex in original_examples]
    augmented_paths = [ex.video_path for ex in augmented_examples if "_aug_" in ex.video_path]
    
    logger.info(f"Original video paths: {original_paths}")
    logger.info(f"Sample augmented paths: {augmented_paths[:5]}")
    
    # Verify labels and metadata preservation
    logger.info("Verifying label and metadata preservation...")
    for i, original in enumerate(original_examples):
        # Find corresponding augmented examples
        augmented_for_original = [
            ex for ex in augmented_examples 
            if ex.video_path.startswith(original.video_path + "_aug_")
        ]
        
        logger.info(f"Original {i}: {original.ground_truth_label} -> "
                   f"{len(augmented_for_original)} augmented versions")
        
        # Check that labels are preserved
        for aug_ex in augmented_for_original:
            assert aug_ex.ground_truth_label == original.ground_truth_label
            assert aug_ex.severity_score == original.severity_score
            assert aug_ex.metadata == original.metadata
    
    logger.info("All labels and metadata correctly preserved!")
    
    # Demonstrate individual augmentation techniques
    logger.info("Demonstrating individual augmentation techniques...")
    sample_example = original_examples[0]
    
    # Test spatial augmentations
    logger.info("Testing spatial augmentations...")
    for aug_idx in range(3):
        augmented = data_augmentation._augment_single_example(sample_example, aug_idx)
        original_coords = sample_example.pose_sequence.keypoints[0][0]
        augmented_coords = augmented.pose_sequence.keypoints[0][0]
        
        logger.info(f"  Aug {aug_idx}: Original ({original_coords.x:.2f}, {original_coords.y:.2f}) -> "
                   f"Augmented ({augmented_coords.x:.2f}, {augmented_coords.y:.2f})")
    
    logger.info("=== Demo Complete ===")


if __name__ == "__main__":
    demonstrate_augmentation_pipeline()