#!/usr/bin/env python3
"""
Verify data directory structure and contents.

Checks if all required datasets are present and properly structured.
"""

import os
from pathlib import Path
from collections import defaultdict


def get_size_gb(path):
    """Calculate total size of directory in GB."""
    total = 0
    for entry in Path(path).rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total / (1024 ** 3)


def count_files_by_extension(directory, extensions):
    """Count files by extension in a directory."""
    counts = defaultdict(int)
    for ext in extensions:
        pattern = f"**/*{ext}"
        counts[ext] = len(list(Path(directory).glob(pattern)))
    return counts


def verify_data_structure():
    """Verify the data directory structure and contents."""
    
    print("=" * 60)
    print("Data Verification Report")
    print("=" * 60)
    
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("\n❌ ERROR: 'data/' directory not found!")
        print("\nPlease run: python scripts/download_datasets.py")
        return False
    
    # Expected directories
    expected_dirs = {
        "data/raw": ["Raw video files"],
        "data/processed": ["Processed features (PKL, H5, NPY)"],
        "data/pose_data": ["Extracted pose keypoints"],
        "data/external": ["External datasets (optional)"]
    }
    
    all_ok = True
    
    print("\n📁 Directory Structure:")
    print("-" * 60)
    
    for dir_path, description in expected_dirs.items():
        path = Path(dir_path)
        exists = path.exists()
        status = "✓" if exists else "✗"
        optional = "(optional)" in description[0].lower()
        
        print(f"{status} {dir_path:30} ", end="")
        
        if exists:
            size_gb = get_size_gb(path)
            file_count = len(list(path.rglob('*')))
            print(f"{size_gb:6.2f} GB  ({file_count} files)")
        else:
            if not optional:
                print("MISSING")
                all_ok = False
            else:
                print("Not present (optional)")
    
    # Check raw videos
    print("\n📹 Raw Videos:")
    print("-" * 60)
    
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        video_exts = ['.mp4', '.avi', '.mov', '.mkv']
        video_counts = count_files_by_extension(raw_dir, video_exts)
        total_videos = sum(video_counts.values())
        
        for ext, count in video_counts.items():
            if count > 0:
                print(f"  {ext:10} {count:5} files")
        
        print(f"\n  Total:     {total_videos:5} video files")
        
        if total_videos == 0:
            print("\n  ⚠️  Warning: No video files found in data/raw/")
            all_ok = False
    else:
        print("  ❌ data/raw/ directory not found")
        all_ok = False
    
    # Check processed data
    print("\n🔧 Processed Data:")
    print("-" * 60)
    
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        data_exts = ['.pkl', '.h5', '.npy', '.npz', '.csv']
        data_counts = count_files_by_extension(processed_dir, data_exts)
        total_processed = sum(data_counts.values())
        
        for ext, count in data_counts.items():
            if count > 0:
                print(f"  {ext:10} {count:5} files")
        
        print(f"\n  Total:     {total_processed:5} processed files")
        
        if total_processed == 0:
            print("\n  ⚠️  Warning: No processed files found")
            print("      Run preprocessing: python scripts/preprocess_data.py")
    else:
        print("  ❌ data/processed/ directory not found")
        all_ok = False
    
    # Overall summary
    print("\n" + "=" * 60)
    
    total_size = get_size_gb(data_dir)
    print(f"Total data size: {total_size:.2f} GB")
    
    if all_ok:
        print("\n✅ Data verification PASSED")
        print("\nYou can now:")
        print("  - Train models: python scripts/train_model.py")
        print("  - Run analysis: python web/app.py")
    else:
        print("\n❌ Data verification FAILED")
        print("\nPlease:")
        print("  - Download datasets: python scripts/download_datasets.py")
        print("  - Or check DATASET_SETUP.md for manual setup")
    
    print("=" * 60)
    
    return all_ok


if __name__ == "__main__":
    import sys
    success = verify_data_structure()
    sys.exit(0 if success else 1)
