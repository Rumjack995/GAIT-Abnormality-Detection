"""
Download GAVD videos from YouTube URLs in the annotation files.

This script parses the GAVD clinical annotations and downloads the unique
YouTube videos using yt-dlp.
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_ytdlp_installed():
    """Check if yt-dlp is installed."""
    try:
        import yt_dlp
        print(f"[OK] yt-dlp version: {yt_dlp.version.__version__}")
        return True
    except ImportError:
        return False


def install_ytdlp():
    """Install yt-dlp using pip."""
    print("[INSTALL] Installing yt-dlp...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'yt-dlp'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install yt-dlp: {e}")
        return False


def parse_annotations(raw_data_dir):
    """
    Parse all GAVD annotation CSV files to extract unique video URLs.
    
    Returns:
        dict: {gait_pattern: [(video_id, url), ...]}
    """
    print("\n[PARSE] Parsing GAVD annotation files...")
    
    videos_by_pattern = defaultdict(set)
    
    # Find all CSV files
    csv_files = list(raw_data_dir.glob('GAVD_Clinical_Annotations_*.csv'))
    
    if not csv_files:
        print("[ERROR] No GAVD annotation files found!")
        print(f"  Expected location: {raw_data_dir}")
        return {}
    
    print(f"[OK] Found {len(csv_files)} annotation files")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Group by video ID and gait pattern
            for _, row in df.drop_duplicates(subset=['id', 'gait_pat']).iterrows():
                video_id = row.get('id')
                url = row.get('url')
                gait_pattern = row.get('gait_pat', 'unknown')
                
                if video_id and url:
                    videos_by_pattern[gait_pattern].add((video_id, url))
                    
        except Exception as e:
            print(f"[WARNING] Error reading {csv_file.name}: {e}")
    
    # Print summary
    print("\n[INFO] Videos by gait pattern:")
    total = 0
    for pattern, videos in sorted(videos_by_pattern.items()):
        count = len(videos)
        total += count
        print(f"  {pattern}: {count} videos")
    print(f"  Total unique videos: {total}")
    
    return videos_by_pattern


def map_pattern_to_folder(pattern):
    """Map GAVD gait pattern names to project folder names."""
    pattern_lower = pattern.lower() if pattern else 'unknown'
    
    if 'normal' in pattern_lower:
        return 'normal'
    elif 'parkin' in pattern_lower:
        return 'parkinsonian'
    elif 'hemi' in pattern_lower:
        return 'hemiplegic'
    elif 'atax' in pattern_lower:
        return 'ataxic'
    else:
        return 'other_abnormal'


def download_video(video_id, url, output_path, max_retries=2):
    """
    Download a single video using yt-dlp Python API.
    
    Args:
        video_id: YouTube video ID
        url: YouTube URL
        output_path: Path to save the video
        max_retries: Number of retry attempts
        
    Returns:
        bool: True if successful
    """
    import yt_dlp
    
    # Check if already downloaded
    existing = list(output_path.glob(f"{video_id}.*"))
    if existing:
        return True  # Already downloaded
    
    output_template = str(output_path / f"{video_id}.%(ext)s")
    
    ydl_opts = {
        'format': 'best[height<=720]',  # Max 720p to save space
        'outtmpl': output_template,
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 30,
        'retries': max_retries,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        # Video might be unavailable, private, or region-locked
        return False


def download_all_videos(videos_by_pattern, raw_data_dir, max_per_category=None):
    """
    Download all videos organized by gait pattern.
    
    Args:
        videos_by_pattern: Dict mapping patterns to video tuples
        raw_data_dir: Base output directory
        max_per_category: Optional limit per category (for testing)
    """
    print("\n" + "=" * 60)
    print("Downloading GAVD Videos")
    print("=" * 60)
    
    if max_per_category:
        print(f"[NOTE] Limiting to {max_per_category} videos per category")
    
    stats = {'success': 0, 'failed': 0, 'skipped': 0}
    failed_videos = []
    
    for pattern, videos in sorted(videos_by_pattern.items()):
        folder = map_pattern_to_folder(pattern)
        output_dir = raw_data_dir / folder
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[CATEGORY] {pattern} -> {folder}/")
        
        videos_list = list(videos)
        if max_per_category:
            videos_list = videos_list[:max_per_category]
        
        for i, (video_id, url) in enumerate(videos_list, 1):
            print(f"  [{i}/{len(videos_list)}] Downloading {video_id}...", end=' ')
            
            success = download_video(video_id, url, output_dir)
            
            if success:
                # Check if was skipped or new download
                existing = list(output_dir.glob(f"{video_id}.*"))
                if existing:
                    print("[OK]")
                    stats['success'] += 1
                else:
                    print("[FAIL - no output]")
                    stats['failed'] += 1
                    failed_videos.append((video_id, url, pattern))
            else:
                print("[FAIL]")
                stats['failed'] += 1
                failed_videos.append((video_id, url, pattern))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"  Successful: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
    
    if failed_videos:
        print(f"\n[WARNING] {len(failed_videos)} videos failed to download")
        print("  These may be unavailable, private, or region-locked.")
        
        # Save failed list for manual retry
        failed_file = raw_data_dir / 'failed_downloads.txt'
        with open(failed_file, 'w') as f:
            for vid, url, pattern in failed_videos:
                f.write(f"{vid}\t{url}\t{pattern}\n")
        print(f"  Failed list saved to: {failed_file}")
    
    return stats


def count_downloaded_videos(raw_data_dir):
    """Count videos already downloaded in each category."""
    print("\n[INFO] Current video counts:")
    total = 0
    for folder in ['normal', 'hemiplegic', 'parkinsonian', 'ataxic', 'other_abnormal']:
        folder_path = raw_data_dir / folder
        if folder_path.exists():
            videos = list(folder_path.glob('*.mp4')) + \
                    list(folder_path.glob('*.webm')) + \
                    list(folder_path.glob('*.mkv'))
            count = len(videos)
            total += count
            if count > 0:
                print(f"  {folder}: {count} videos")
    print(f"  Total: {total} videos")
    return total


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download GAVD videos from YouTube')
    parser.add_argument('--max', type=int, default=None,
                       help='Max videos to download per category (for testing)')
    parser.add_argument('--count-only', action='store_true',
                       help='Only count existing videos, do not download')
    args = parser.parse_args()
    
    # Setup paths
    raw_data_dir = project_root / "data" / "raw"
    
    print("\n" + "=" * 60)
    print("GAVD YouTube Video Downloader")
    print("=" * 60)
    print(f"Data directory: {raw_data_dir}")
    
    # Count existing videos
    existing_count = count_downloaded_videos(raw_data_dir)
    
    if args.count_only:
        return
    
    # Check/install yt-dlp
    if not check_ytdlp_installed():
        print("\n[WARNING] yt-dlp is not installed.")
        if not install_ytdlp():
            print("\n[ERROR] Could not install yt-dlp. Please install manually:")
            print("  pip install yt-dlp")
            return
        
        # Verify installation
        if not check_ytdlp_installed():
            print("[ERROR] yt-dlp installation failed")
            return
    
    # Parse annotations
    videos_by_pattern = parse_annotations(raw_data_dir)
    
    if not videos_by_pattern:
        print("\n[ERROR] No videos found in annotations!")
        print("Make sure you ran download_gavd_no_git.py first.")
        return
    
    # Download videos
    print("\n[INFO] Starting downloads...")
    print("[NOTE] This may take several hours depending on your internet speed.")
    print("[NOTE] Videos are downloaded at max 720p to save space.")
    print("[NOTE] Press Ctrl+C to stop at any time.")
    
    try:
        stats = download_all_videos(
            videos_by_pattern, 
            raw_data_dir,
            max_per_category=args.max
        )
    except KeyboardInterrupt:
        print("\n\n[STOPPED] Download interrupted by user.")
    
    # Final count
    count_downloaded_videos(raw_data_dir)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Download Complete!")
    print("=" * 60)
    print("\nNext step: Run preprocessing script")
    print("  python scripts/preprocess_dataset.py")


if __name__ == "__main__":
    main()
