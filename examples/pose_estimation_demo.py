"""
Demo script for pose estimation functionality.

This script demonstrates how to use the PoseEstimator class to extract
and track human pose keypoints from video frames.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from gait_analysis.pose_estimation import PoseEstimator
from gait_analysis.utils.data_structures import PoseSequence


def create_demo_video_frames(num_frames: int = 30) -> list:
    """
    Create demo video frames with a simple animated stick figure.
    
    Args:
        num_frames: Number of frames to generate
        
    Returns:
        List of video frames as numpy arrays
    """
    frames = []
    width, height = 640, 480
    
    for frame_idx in range(num_frames):
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Animate the stick figure (simple walking motion)
        center_x = width // 2
        center_y = height // 2
        
        # Add some animation to simulate walking
        leg_offset = int(20 * np.sin(frame_idx * 0.3))  # Leg swing
        arm_offset = int(15 * np.cos(frame_idx * 0.3))  # Arm swing
        
        # Head
        cv2.circle(frame, (center_x, center_y - 100), 25, (255, 255, 255), -1)
        
        # Body
        cv2.line(frame, (center_x, center_y - 75), (center_x, center_y + 50), (255, 255, 255), 4)
        
        # Arms (animated)
        cv2.line(frame, (center_x, center_y - 30), 
                (center_x - 40 + arm_offset, center_y + 10), (255, 255, 255), 3)
        cv2.line(frame, (center_x, center_y - 30), 
                (center_x + 40 - arm_offset, center_y + 10), (255, 255, 255), 3)
        
        # Legs (animated)
        cv2.line(frame, (center_x, center_y + 50), 
                (center_x - 25 + leg_offset, center_y + 120), (255, 255, 255), 3)
        cv2.line(frame, (center_x, center_y + 50), 
                (center_x + 25 - leg_offset, center_y + 120), (255, 255, 255), 3)
        
        frames.append(frame)
    
    return frames


def demonstrate_pose_extraction():
    """Demonstrate basic pose extraction functionality."""
    print("=== Pose Extraction Demo ===")
    
    # Initialize pose estimator
    pose_estimator = PoseEstimator(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    
    # Create demo frames
    print("Creating demo video frames...")
    frames = create_demo_video_frames(num_frames=60)
    print(f"Created {len(frames)} frames")
    
    # Extract poses
    print("Extracting poses from frames...")
    pose_sequence = pose_estimator.extract_poses(frames)
    
    # Display results
    print(f"Extracted poses from {len(pose_sequence.keypoints)} frames")
    print(f"Average confidence: {np.mean(pose_sequence.confidence_scores):.3f}")
    
    # Validate sequence
    is_valid = pose_estimator.validate_pose_sequence(pose_sequence)
    print(f"Pose sequence is valid: {is_valid}")
    
    return pose_sequence, pose_estimator


def demonstrate_pose_tracking(pose_sequence: PoseSequence, pose_estimator: PoseEstimator):
    """Demonstrate pose tracking and temporal consistency."""
    print("\n=== Pose Tracking Demo ===")
    
    # Apply tracking
    print("Applying temporal tracking...")
    tracked_pose = pose_estimator.track_landmarks(pose_sequence)
    
    print(f"Tracking ID: {tracked_pose.tracking_id}")
    print(f"Tracking confidence: {tracked_pose.tracking_confidence:.3f}")
    
    # Get quality metrics
    quality_metrics = pose_estimator.get_pose_quality_metrics(tracked_pose)
    
    print("\nPose Quality Metrics:")
    for metric_name, value in quality_metrics.items():
        print(f"  {metric_name}: {value:.3f}")
    
    return tracked_pose


def demonstrate_pose_filtering(pose_sequence: PoseSequence, pose_estimator: PoseEstimator):
    """Demonstrate pose filtering functionality."""
    print("\n=== Pose Filtering Demo ===")
    
    # Filter low confidence keypoints
    print("Filtering low confidence keypoints...")
    filtered_sequence = pose_estimator.filter_low_confidence_keypoints(
        pose_sequence, min_confidence=0.3
    )
    
    # Compare original vs filtered
    original_confident = sum(1 for score in pose_sequence.confidence_scores if score > 0.3)
    filtered_confident = sum(1 for score in filtered_sequence.confidence_scores if score > 0.3)
    
    print(f"Original confident frames: {original_confident}/{len(pose_sequence.confidence_scores)}")
    print(f"Filtered confident frames: {filtered_confident}/{len(filtered_sequence.confidence_scores)}")
    
    return filtered_sequence


def visualize_confidence_over_time(pose_sequence: PoseSequence, tracked_pose):
    """Visualize confidence scores over time."""
    print("\n=== Confidence Visualization ===")
    
    try:
        plt.figure(figsize=(12, 6))
        
        # Plot original confidence
        plt.subplot(1, 2, 1)
        plt.plot(pose_sequence.timestamps, pose_sequence.confidence_scores, 'b-', label='Original')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Confidence Score')
        plt.title('Original Pose Confidence')
        plt.grid(True)
        plt.legend()
        
        # Plot tracked confidence
        plt.subplot(1, 2, 2)
        tracked_sequence = tracked_pose.pose_sequence
        plt.plot(tracked_sequence.timestamps, tracked_sequence.confidence_scores, 'r-', label='Tracked')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Confidence Score')
        plt.title('Tracked Pose Confidence')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('pose_confidence_comparison.png', dpi=150, bbox_inches='tight')
        print("Confidence visualization saved as 'pose_confidence_comparison.png'")
        
    except Exception as e:
        print(f"Could not create visualization: {e}")


def main():
    """Main demo function."""
    print("Gait Analysis - Pose Estimation Demo")
    print("=" * 40)
    
    try:
        # Demonstrate pose extraction
        pose_sequence, pose_estimator = demonstrate_pose_extraction()
        
        # Demonstrate pose tracking
        tracked_pose = demonstrate_pose_tracking(pose_sequence, pose_estimator)
        
        # Demonstrate pose filtering
        filtered_sequence = demonstrate_pose_filtering(pose_sequence, pose_estimator)
        
        # Visualize results
        visualize_confidence_over_time(pose_sequence, tracked_pose)
        
        print("\n=== Demo Summary ===")
        print("✓ Pose extraction from video frames")
        print("✓ Temporal consistency and tracking")
        print("✓ Pose sequence validation")
        print("✓ Quality metrics calculation")
        print("✓ Low confidence keypoint filtering")
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if 'pose_estimator' in locals():
            del pose_estimator


if __name__ == "__main__":
    main()