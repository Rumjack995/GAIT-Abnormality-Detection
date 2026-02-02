"""
Download GAVD dataset without Git (uses GitHub ZIP download)

This is an alternative to download_gavd.py for systems without Git installed.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
import requests
from io import BytesIO

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_zip(url, dest_dir):
    """
    Download and extract a ZIP file from URL.
    
    Args:
        url: URL to download from
        dest_dir: Directory to extract to
        
    Returns:
        Path to extracted directory or None on failure
    """
    print(f"[DOWNLOAD] Downloading from: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"[INFO] Download size: {total_size / 1024 / 1024:.1f} MB")
        
        # Download to memory
        content = BytesIO()
        downloaded = 0
        chunk_size = 8192
        
        for chunk in response.iter_content(chunk_size=chunk_size):
            content.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\r[PROGRESS] {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end='')
        
        print()  # New line after progress
        
        # Extract ZIP
        print("[EXTRACT] Extracting files...")
        content.seek(0)
        
        with zipfile.ZipFile(content) as zf:
            zf.extractall(dest_dir)
        
        # Find extracted directory (usually has -main or -master suffix)
        extracted_dirs = [d for d in dest_dir.iterdir() if d.is_dir()]
        if extracted_dirs:
            return extracted_dirs[0]
        return dest_dir
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Download failed: {e}")
        return None
    except zipfile.BadZipFile:
        print("[ERROR] Invalid ZIP file downloaded")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return None


def search_github_for_gavd():
    """
    Search GitHub API for GAVD repository.
    
    Returns:
        List of potential repository URLs
    """
    print("\n[SEARCH] Searching GitHub for GAVD repository...")
    
    try:
        # Search GitHub API
        search_url = "https://api.github.com/search/repositories?q=GAVD+gait+abnormality"
        response = requests.get(search_url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            repos = []
            
            for item in data.get('items', [])[:5]:
                repos.append({
                    'name': item['full_name'],
                    'url': item['html_url'],
                    'zip_url': f"https://github.com/{item['full_name']}/archive/refs/heads/main.zip",
                    'description': item.get('description', 'No description'),
                    'stars': item.get('stargazers_count', 0)
                })
            
            return repos
    except Exception as e:
        print(f"[WARNING] GitHub search failed: {e}")
    
    return []


def organize_dataset(source_dir, raw_data_dir):
    """
    Organize GAVD dataset into project structure.
    
    Args:
        source_dir: Path to downloaded GAVD data
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
    
    # Look for videos in source directory
    print(f"\n[SEARCH] Searching for videos in {source_dir}...")
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos_found = []
    
    for ext in video_extensions:
        videos_found.extend(source_dir.rglob(f'*{ext}'))
    
    if videos_found:
        print(f"[OK] Found {len(videos_found)} videos")
        
        for video_path in videos_found:
            video_name_lower = video_path.name.lower()
            parent_lower = video_path.parent.name.lower()
            
            # Determine category from path or filename
            if 'normal' in video_name_lower or 'normal' in parent_lower:
                dest_dir = categories['normal']
            elif 'hemi' in video_name_lower or 'hemi' in parent_lower:
                dest_dir = categories['hemiplegic']
            elif 'parkin' in video_name_lower or 'parkin' in parent_lower:
                dest_dir = categories['parkinsonian']
            elif 'atax' in video_name_lower or 'atax' in parent_lower:
                dest_dir = categories['ataxic']
            else:
                dest_dir = categories['other_abnormal']
            
            dest_path = dest_dir / video_path.name
            shutil.copy(video_path, dest_path)
            print(f"  Copied: {video_path.name} -> {dest_dir.name}/")
    else:
        print("\n[WARNING] No videos found in downloaded repository.")
        print("\nThe GAVD repository likely contains:")
        print("  - Video URLs/links (not actual video files)")
        print("  - Annotations and metadata")
        print("  - Download scripts")
        
        # Look for useful files
        print("\n[INFO] Looking for useful files...")
        
        for pattern in ['*.csv', '*.json', '*.txt', '*.md', '*.py']:
            for file_path in source_dir.rglob(pattern):
                print(f"  Found: {file_path.relative_to(source_dir)}")
        
        # Look for annotation files and copy them
        annotation_files = list(source_dir.rglob('*.csv')) + list(source_dir.rglob('*.json'))
        if annotation_files:
            print(f"\n[INFO] Copying annotation files to data directory...")
            for ann_file in annotation_files:
                dest = raw_data_dir / ann_file.name
                shutil.copy(ann_file, dest)
                print(f"  Copied: {ann_file.name}")


def print_instructions():
    """Print instructions for manual download."""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    
    print("""
The GAVD dataset may require manual download. Here's what to do:

1. SEARCH FOR GAVD:
   - Go to: https://github.com/search?q=GAVD+gait+abnormality
   - Or search: "GAVD gait abnormality video dataset"

2. DOWNLOAD VIDEOS:
   - The repository may contain a list of video URLs
   - Download videos manually or use any provided download script
   - Place videos in the appropriate folders:

     data/raw/
     ├── normal/         <- Normal gait videos
     ├── hemiplegic/     <- Hemiplegic gait videos
     ├── parkinsonian/   <- Parkinsonian gait videos
     ├── ataxic/         <- Ataxic gait videos
     └── other_abnormal/ <- Other abnormal gait videos

3. ALTERNATIVE DATASETS:
   If GAVD is unavailable, try these alternatives:

   a) CMU Abnormal Gait Dataset (AGD-CMU)
      - 89 normal + 49 abnormal gaits
      - Search: "CMU abnormal gait dataset"

   b) Create your own from YouTube:
      - Search: "hemiplegic gait clinical", "parkinsonian gait", etc.
      - Download with yt-dlp: pip install yt-dlp

4. AFTER PLACING VIDEOS:
   Run preprocessing:
   python scripts/preprocess_dataset.py
""")


def print_summary(raw_data_dir):
    """Print summary of available data."""
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    
    total_videos = 0
    for category_dir in raw_data_dir.iterdir():
        if category_dir.is_dir() and not category_dir.name.startswith('.'):
            videos = list(category_dir.glob('*.mp4')) + \
                    list(category_dir.glob('*.avi')) + \
                    list(category_dir.glob('*.mov'))
            count = len(videos)
            total_videos += count
            if count > 0:
                print(f"  {category_dir.name}: {count} videos")
    
    print(f"\n  Total: {total_videos} videos")
    
    if total_videos == 0:
        print("\n[WARNING] No videos found yet.")
        print_instructions()
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
    
    print("\n" + "=" * 60)
    print("GAVD Dataset Download (No Git Required)")
    print("=" * 60)
    print(f"Project: {project_root}")
    print(f"Data directory: {raw_data_dir}")
    
    # Create directories
    temp_dir.mkdir(parents=True, exist_ok=True)
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to find GAVD repository
    repos = search_github_for_gavd()
    
    downloaded_dir = None
    
    if repos:
        print(f"\n[INFO] Found {len(repos)} potential repositories:")
        for i, repo in enumerate(repos, 1):
            print(f"  {i}. {repo['name']} (Stars: {repo['stars']})")
            print(f"     {repo['description'][:60]}...")
        
        # Try to download the most relevant one
        for repo in repos:
            print(f"\n[TRY] Attempting to download: {repo['name']}")
            
            # Try main branch
            downloaded_dir = download_zip(repo['zip_url'], temp_dir)
            
            if not downloaded_dir:
                # Try master branch
                master_url = repo['zip_url'].replace('/main.zip', '/master.zip')
                downloaded_dir = download_zip(master_url, temp_dir)
            
            if downloaded_dir:
                print(f"[SUCCESS] Downloaded: {repo['name']}")
                break
    
    if downloaded_dir and downloaded_dir.exists():
        # Organize dataset
        organize_dataset(downloaded_dir, raw_data_dir)
    else:
        print("\n[WARNING] Could not automatically download GAVD repository.")
    
    # Print summary
    print_summary(raw_data_dir)
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
