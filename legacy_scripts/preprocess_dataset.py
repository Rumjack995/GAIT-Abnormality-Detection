"""
Preprocess GAVD dataset for training

This script:
1. Validates all videos
2. Extracts frames
3. Runs pose estimation
4. Saves processed data for training
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gait_analysis.video_processing import VideoProcessor
from gait_analysis.pose_estimation import PoseEstimator


def validate_videos(raw_data_dir):
    """
    Validate all videos in the dataset.
    
    Args:
        raw_data_dir: Path to raw data directory
        
    Returns:
        Dictionary of valid videos by category
    """
    print("=" * 60)
    print("Video Validation")
    print("=" * 60)
    
    processor = VideoProcessor()
    valid_videos = {}
    invalid_videos = []
    
    for category_dir in raw_data_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        
        category = category_dir.name
        valid_videos[category] = []
        
        # Find all videos
        videos = list(category_dir.glob('*.mp4')) + \
                list(category_dir.glob('*.avi')) + \
                list(category_dir.glob('*.mov'))
        
        print(f"\n[FOLDER] {category}: {len(videos)} videos")
        
        for video_path in tqdm(videos, desc=f"Validating {category}"):
            result = processor.validate_video(str(video_path))
            
            if result.is_valid:
                valid_videos[category].append(video_path)
            else:
                invalid_videos.append({
                    'path': str(video_path),
                    'category': category,
                    'error': result.error_message
                })
                print(f"  [X] Invalid: {video_path.name}")
                print(f"     Error: {result.error_message}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    total_valid = sum(len(videos) for videos in valid_videos.values())
    total_invalid = len(invalid_videos)
    
    for category, videos in valid_videos.items():
        print(f"  {category}: {len(videos)} valid videos")
    
    print(f"\n  Total valid: {total_valid}")
    print(f"  Total invalid: {total_invalid}")
    
    # Save invalid videos log
    if invalid_videos:
        invalid_log = raw_data_dir / "invalid_videos.json"
        with open(invalid_log, 'w') as f:
            json.dump(invalid_videos, f, indent=2)
        print(f"\n  Invalid videos logged to: {invalid_log}")
    
    return valid_videos


def process_videos(valid_videos, processed_dir):
    """
    Process valid videos: extract frames and poses.
    
    Args:
        valid_videos: Dictionary of valid videos by category
        processed_dir: Directory to save processed data
    """
    print("\n" + "=" * 60)
    print("Video Processing")
    print("=" * 60)
    
    processor = VideoProcessor()
    estimator = PoseEstimator(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    
    # Create processed directory structure
    for category in valid_videos.keys():
        (processed_dir / category).mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    failed_count = 0
    
    for category, videos in valid_videos.items():
        print(f"\n[FOLDER] Processing {category}: {len(videos)} videos")
        
        for video_path in tqdm(videos, desc=f"Processing {category}"):
            try:
                # Extract frames
                frames = processor.extract_frames_with_enhancement(
                    str(video_path),
                    target_fps=30,
                    enhance_quality=True
                )
                
                if len(frames) == 0:
                    print(f"  [!] No frames extracted: {video_path.name}")
                    failed_count += 1
                    continue
                
                # Extract poses
                pose_sequence = estimator.extract_poses(frames)
                
                # Validate pose sequence
                if not estimator.validate_pose_sequence(pose_sequence):
                    print(f"  [!] Invalid pose sequence: {video_path.name}")
                    failed_count += 1
                    continue
                
                # Apply tracking
                tracked_pose = estimator.track_landmarks(pose_sequence)
                
                # Save processed data
                output_file = processed_dir / category / f"{video_path.stem}.pkl"
                
                processed_data = {
                    'video_name': video_path.name,
                    'category': category,
                    'frames': frames,
                    'pose_sequence': pose_sequence,
                    'tracked_pose': tracked_pose,
                    'num_frames': len(frames),
                    'fps': 30
                }
                
                with open(output_file, 'wb') as f:
                    pickle.dump(processed_data, f)
                
                processed_count += 1
                
            except Exception as e:
                print(f"  [X] Error processing {video_path.name}: {e}")
                failed_count += 1
                continue
    
    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    print(f"  Successfully processed: {processed_count}")
    print(f"  Failed: {failed_count}")
    
    return processed_count


def create_dataset_splits(processed_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Create train/validation/test splits.
    
    Args:
        processed_dir: Directory with processed data
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
    """
    print("\n" + "=" * 60)
    print("Creating Dataset Splits")
    print("=" * 60)
    
    splits = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for category_dir in processed_dir.iterdir():
        if not category_dir.is_dir():
            continue
        
        category = category_dir.name
        processed_files = list(category_dir.glob('*.pkl'))
        
        # Shuffle files
        np.random.seed(42)
        indices = np.random.permutation(len(processed_files))
        
        # Calculate split sizes
        n_train = int(len(processed_files) * train_ratio)
        n_val = int(len(processed_files) * val_ratio)
        
        # Split files
        train_files = [processed_files[i] for i in indices[:n_train]]
        val_files = [processed_files[i] for i in indices[n_train:n_train+n_val]]
        test_files = [processed_files[i] for i in indices[n_train+n_val:]]
        
        splits['train'].extend([{'path': str(f), 'category': category} for f in train_files])
        splits['val'].extend([{'path': str(f), 'category': category} for f in val_files])
        splits['test'].extend([{'path': str(f), 'category': category} for f in test_files])
        
        print(f"\n{category}:")
        print(f"  Train: {len(train_files)}")
        print(f"  Val: {len(val_files)}")
        print(f"  Test: {len(test_files)}")
    
    # Save splits
    splits_file = processed_dir / 'dataset_splits.json'
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n[OK] Splits saved to: {splits_file}")
    
    print("\n" + "=" * 60)
    print("Overall Split Summary")
    print("=" * 60)
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val: {len(splits['val'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    print(f"  Total: {len(splits['train']) + len(splits['val']) + len(splits['test'])} samples")


def main():
    """Main execution function."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    
    print("\n=== GAVD Dataset Preprocessing ===")
    print(f"Raw data: {raw_data_dir}")
    print(f"Processed data: {processed_dir}")
    
    # Create processed directory
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Validate videos
    valid_videos = validate_videos(raw_data_dir)
    
    total_valid = sum(len(videos) for videos in valid_videos.values())
    if total_valid == 0:
        print("\n[X] No valid videos found!")
        print("Please run download_gavd.py first to download the dataset.")
        return
    
    # Step 2: Process videos
    processed_count = process_videos(valid_videos, processed_dir)
    
    if processed_count == 0:
        print("\n[X] No videos were successfully processed!")
        return
    
    # Step 3: Create splits
    create_dataset_splits(processed_dir)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Preprocessing Complete!")
    print("=" * 60)
    print("\nNext step: Train models")
    print("  python scripts/train_models.py")


if __name__ == "__main__":
    main()
