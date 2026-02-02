"""
Download and organize GAVD (Gait Abnormality in Video Dataset)

This script downloads the GAVD dataset from GitHub and organizes it
into the project's data structure.
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_git_installed():
    """Check if git is installed."""
    try:
        subprocess.run(['git', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_gavd_repository(temp_dir):
    """
    Download GAVD repository from GitHub.
    
    Args:
        temp_dir: Temporary directory for download
        
    Returns:
        Path to downloaded repository
    """
    print("=" * 60)
    print("GAVD Dataset Download")
    print("=" * 60)
    
    # Check if git is installed
    if not check_git_installed():
        print("\n[ERROR] Git is not installed!")
        print("\nPlease install Git:")
        print("  1. Download from: https://git-scm.com/download/win")
        print("  2. Install with default settings")
        print("  3. Restart your terminal")
        print("  4. Run this script again")
        return None
    
    # Create temp directory
    temp_dir.mkdir(parents=True, exist_ok=True)
    gavd_dir = temp_dir / "GAVD"
    
    # Clone repository
    print(f"\n[DOWNLOAD] Downloading GAVD repository...")
    print(f"Destination: {gavd_dir}")
    
    try:
        # Try the most likely repository URLs
        repo_urls = [
            "https://github.com/GaitAbnormality/GAVD.git",
            "https://github.com/GAVD/GAVD.git",
            "https://github.com/gait-abnormality/GAVD.git"
        ]
        
        cloned = False
        for repo_url in repo_urls:
            try:
                print(f"\nTrying: {repo_url}")
                subprocess.run(
                    ['git', 'clone', repo_url, str(gavd_dir)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                cloned = True
                print(f"[SUCCESS] Successfully cloned from {repo_url}")
                break
            except subprocess.CalledProcessError:
                continue
        
        if not cloned:
            print("\n[WARNING] Could not find GAVD repository at expected URLs.")
            print("\nPlease manually:")
            print("  1. Search GitHub for 'GAVD gait abnormality dataset'")
            print("  2. Clone the repository to:", gavd_dir)
            print("  3. Run this script again")
            return None
            
        return gavd_dir
        
    except Exception as e:
        print(f"\n[ERROR] Error downloading repository: {e}")
        return None


def organize_dataset(gavd_dir, raw_data_dir):
    """
    Organize GAVD dataset into project structure.
    
    Args:
        gavd_dir: Path to GAVD repository
        raw_data_dir: Path to project's raw data directory
    """
    print("\n" + "=" * 60)
    print("Organizing Dataset")
    print("=" * 60)
    
    # Create category directories
    categories = {
        'normal': raw_data_dir / 'normal',
        'hemiplegic': raw_data_dir / 'hemiplegic',
        'parkinsonian': raw_data_dir / 'parkinsonian',
        'ataxic': raw_data_dir / 'ataxic',
        'other_abnormal': raw_data_dir / 'other_abnormal'
    }
    
    for category, path in categories.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created directory: {category}/")
    
    # Look for videos in GAVD directory
    print(f"\n[SEARCH] Searching for videos in {gavd_dir}...")
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos_found = []
    
    for ext in video_extensions:
        videos_found.extend(gavd_dir.rglob(f'*{ext}'))
    
    if videos_found:
        print(f"[OK] Found {len(videos_found)} videos")
        
        # Copy videos (basic organization - adjust based on actual GAVD structure)
        for video_path in videos_found:
            # Try to determine category from path or filename
            video_name_lower = video_path.name.lower()
            
            if 'normal' in video_name_lower:
                dest_dir = categories['normal']
            elif 'hemi' in video_name_lower:
                dest_dir = categories['hemiplegic']
            elif 'parkin' in video_name_lower:
                dest_dir = categories['parkinsonian']
            elif 'atax' in video_name_lower:
                dest_dir = categories['ataxic']
            else:
                dest_dir = categories['other_abnormal']
            
            dest_path = dest_dir / video_path.name
            shutil.copy(video_path, dest_path)
            print(f"  Copied: {video_path.name} -> {dest_dir.name}/")
    else:
        print("\n[WARNING] No videos found in repository.")
        print("\nThe GAVD repository likely contains:")
        print("  - Video URLs/links (not actual video files)")
        print("  - Annotations and metadata")
        print("  - Download scripts")
        
        print("\n📋 Next steps:")
        print(f"  1. Check {gavd_dir} for README or download instructions")
        print("  2. Look for video URL lists or download scripts")
        print("  3. Download videos manually or using provided scripts")
        print(f"  4. Place videos in: {raw_data_dir}")
        
        # Look for annotation files
        annotation_files = list(gavd_dir.rglob('*.csv')) + list(gavd_dir.rglob('*.json'))
        if annotation_files:
            print(f"\n[INFO] Found annotation files:")
            for ann_file in annotation_files:
                print(f"  - {ann_file.name}")
                # Copy to data directory
                shutil.copy(ann_file, raw_data_dir / ann_file.name)
                print(f"    Copied to: {raw_data_dir / ann_file.name}")


def print_summary(raw_data_dir):
    """Print summary of downloaded dataset."""
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    
    total_videos = 0
    for category_dir in raw_data_dir.iterdir():
        if category_dir.is_dir() and category_dir.name != '__pycache__':
            videos = list(category_dir.glob('*.mp4')) + \
                    list(category_dir.glob('*.avi')) + \
                    list(category_dir.glob('*.mov'))
            count = len(videos)
            total_videos += count
            print(f"  {category_dir.name}: {count} videos")
    
    print(f"\n  Total: {total_videos} videos")
    
    if total_videos == 0:
        print("\n[WARNING] No videos downloaded yet.")
        print("Please follow the instructions above to download videos.")
    else:
        print("\n[SUCCESS] Dataset ready for preprocessing!")
        print("\nNext step: Run preprocessing script")
        print("  python scripts/preprocess_dataset.py")


def main():
    """Main execution function."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    temp_dir = project_root / "temp_download"
    raw_data_dir = project_root / "data" / "raw"
    
    print("\n=== GAVD Dataset Setup ===")
    print(f"Project: {project_root}")
    print(f"Data directory: {raw_data_dir}")
    
    # Download repository
    gavd_dir = download_gavd_repository(temp_dir)
    
    if gavd_dir and gavd_dir.exists():
        # Organize dataset
        organize_dataset(gavd_dir, raw_data_dir)
    
    # Print summary
    print_summary(raw_data_dir)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Setup Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
