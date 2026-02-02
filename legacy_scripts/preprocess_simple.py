"""
Simplified preprocessing for GAVD dataset

This script prepares videos for training without MediaPipe dependency:
1. Validates video files
2. Extracts frames using OpenCV only
3. Creates train/val/test splits
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def validate_video_simple(video_path):
    """Simple video validation using OpenCV only."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "Cannot open video"
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Basic validation
        if width < 100 or height < 100:
            return False, f"Resolution too low: {width}x{height}"
        if duration < 1:
            return False, f"Video too short: {duration:.1f}s"
        
        return True, {'width': width, 'height': height, 'fps': fps, 'duration': duration, 'frames': frame_count}
    except Exception as e:
        return False, str(e)


def extract_frames_simple(video_path, target_fps=10, max_frames=300):
    """
    Extract frames from video at target FPS.
    
    Args:
        video_path: Path to video
        target_fps: Target frames per second to extract
        max_frames: Maximum frames to extract
        
    Returns:
        numpy array of frames (N, H, W, C)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(original_fps / target_fps))
    
    frames = []
    frame_idx = 0
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            # Resize to standard size for model input
            frame = cv2.resize(frame, (224, 224))
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        frame_idx += 1
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    return np.array(frames, dtype=np.float32) / 255.0


def process_dataset(raw_data_dir, processed_dir):
    """
    Process all videos in the dataset.
    
    Args:
        raw_data_dir: Path to raw video directories
        processed_dir: Path to save processed data
    """
    print("\n" + "=" * 60)
    print("GAVD Dataset Preprocessing (Simplified)")
    print("=" * 60)
    print(f"Raw data: {raw_data_dir}")
    print(f"Output: {processed_dir}")
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Map folder names to labels
    label_map = {
        'normal': 0,
        'parkinsonian': 1,
        'hemiplegic': 2,
        'ataxic': 3,
        'other_abnormal': 4
    }
    
    all_data = []
    stats = {'valid': 0, 'invalid': 0, 'processed': 0}
    
    # Process each category
    for category_dir in raw_data_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        
        category = category_dir.name
        
        # Get label
        if category in label_map:
            label = label_map[category]
        else:
            label = 4  # other_abnormal
        
        # Find videos
        videos = list(category_dir.glob('*.mp4')) + \
                list(category_dir.glob('*.webm')) + \
                list(category_dir.glob('*.mkv')) + \
                list(category_dir.glob('*.avi'))
        
        if not videos:
            continue
        
        print(f"\n[CATEGORY] {category}: {len(videos)} videos")
        
        for video_path in tqdm(videos, desc=f"Processing {category}"):
            # Validate
            is_valid, info = validate_video_simple(video_path)
            
            if not is_valid:
                stats['invalid'] += 1
                continue
            
            stats['valid'] += 1
            
            # Extract frames
            frames = extract_frames_simple(video_path, target_fps=10, max_frames=150)
            
            if frames is None or len(frames) < 10:
                print(f"  [!] Insufficient frames: {video_path.name}")
                continue
            
            # Save processed data
            output_file = processed_dir / f"{video_path.stem}.pkl"
            
            data = {
                'video_name': video_path.name,
                'category': category,
                'label': label,
                'frames': frames,
                'num_frames': len(frames),
                'video_info': info
            }
            
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
            
            all_data.append({
                'path': str(output_file),
                'category': category,
                'label': label,
                'num_frames': len(frames)
            })
            
            stats['processed'] += 1
    
    return all_data, stats


def create_splits(all_data, processed_dir, train_ratio=0.7, val_ratio=0.15):
    """Create train/val/test splits."""
    print("\n" + "=" * 60)
    print("Creating Dataset Splits")
    print("=" * 60)
    
    np.random.seed(42)
    indices = np.random.permutation(len(all_data))
    
    n_train = int(len(all_data) * train_ratio)
    n_val = int(len(all_data) * val_ratio)
    
    splits = {
        'train': [all_data[i] for i in indices[:n_train]],
        'val': [all_data[i] for i in indices[n_train:n_train + n_val]],
        'test': [all_data[i] for i in indices[n_train + n_val:]]
    }
    
    # Save splits
    splits_file = processed_dir / 'dataset_splits.json'
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val: {len(splits['val'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    print(f"\nSplits saved to: {splits_file}")
    
    return splits


def main():
    """Main execution."""
    raw_data_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    
    # Process videos
    all_data, stats = process_dataset(raw_data_dir, processed_dir)
    
    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    print(f"  Valid videos: {stats['valid']}")
    print(f"  Invalid videos: {stats['invalid']}")
    print(f"  Successfully processed: {stats['processed']}")
    
    if stats['processed'] == 0:
        print("\n[X] No videos processed!")
        print("Please download videos first: python scripts/download_gavd_videos.py")
        return
    
    # Create splits
    splits = create_splits(all_data, processed_dir)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Preprocessing Complete!")
    print("=" * 60)
    print("\nNext step: Train models")
    print("  python scripts/train_models.py")


if __name__ == "__main__":
    main()
