"""
Download additional gait datasets for missing categories.

This script downloads:
1. ROC-HCI Ataxia dataset (GitHub) - for ataxic gait
2. Kaggle Cerebellar Ataxia dataset - for ataxic gait  
3. Additional Parkinson's data
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
import shutil

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_file(url, dest_path, desc="Downloading"):
    """Download a file with progress."""
    print(f"[DOWNLOAD] {desc}")
    print(f"  URL: {url}")
    print(f"  Dest: {dest_path}")
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"\r  Progress: {pct:.1f}%", end='')
        
        print(f"\n  [OK] Downloaded {dest_path.name}")
        return True
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        return False


def download_github_repo_zip(owner, repo, dest_dir):
    """Download a GitHub repository as ZIP."""
    url = f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip"
    zip_path = dest_dir / f"{repo}.zip"
    
    # Try main branch first, then master
    if not download_file(url, zip_path, f"GitHub: {owner}/{repo}"):
        url = f"https://github.com/{owner}/{repo}/archive/refs/heads/master.zip"
        if not download_file(url, zip_path, f"GitHub: {owner}/{repo} (master)"):
            return None
    
    # Extract
    print(f"  [EXTRACT] Extracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dest_dir)
        
        # Find extracted folder
        for item in dest_dir.iterdir():
            if item.is_dir() and item.name.startswith(repo):
                print(f"  [OK] Extracted to {item.name}")
                os.remove(zip_path)
                return item
    except Exception as e:
        print(f"  [ERROR] Extraction failed: {e}")
    
    return None


def download_roc_hci_ataxia(data_dir):
    """Download ROC-HCI Automated Ataxia Gait dataset."""
    print("\n" + "=" * 60)
    print("ROC-HCI Ataxia Dataset")
    print("=" * 60)
    
    dest_dir = data_dir / "external" / "roc_hci_ataxia"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    repo_dir = download_github_repo_zip("ROC-HCI", "Automated-Ataxia-Gait", dest_dir)
    
    if repo_dir:
        # Look for video files
        videos = list(repo_dir.rglob("*.mp4")) + list(repo_dir.rglob("*.avi"))
        print(f"  Found {len(videos)} video files")
        
        # Copy to ataxic folder in raw data
        ataxic_dir = data_dir / "raw" / "ataxic"
        ataxic_dir.mkdir(parents=True, exist_ok=True)
        
        for video in videos[:50]:  # Limit for storage
            shutil.copy(video, ataxic_dir / video.name)
        
        print(f"  [OK] Copied videos to {ataxic_dir}")
        return True
    
    return False


def download_kaggle_ataxia(data_dir):
    """Download Kaggle Cerebellar Ataxia dataset info."""
    print("\n" + "=" * 60)
    print("Kaggle Cerebellar Ataxia Dataset")
    print("=" * 60)
    
    print("  This dataset requires Kaggle API authentication.")
    print("  To download manually:")
    print("  1. Visit: https://www.kaggle.com/datasets/.../gait-analysis-cerebellar-ataxia")
    print("  2. Download the dataset")
    print("  3. Extract to: data/external/kaggle_ataxia/")
    
    # Check if kaggle is available
    try:
        import kaggle
        print("  [OK] Kaggle API available")
        # Would need kaggle.api.dataset_download_files()
    except ImportError:
        print("  [INFO] Install kaggle: pip install kaggle")
    
    return False


def download_mendeley_ataxia(data_dir):
    """Download Mendeley Cerebellar Ataxia dataset."""
    print("\n" + "=" * 60)
    print("Mendeley Cerebellar Ataxia Dataset")
    print("=" * 60)
    
    # Mendeley data requires form submission, provide manual instructions
    print("  This dataset requires manual download from Mendeley Data.")
    print("  URL: https://data.mendeley.com/datasets/ (search: cerebellar ataxia gait)")
    print("  Extract to: data/external/mendeley_ataxia/")
    
    return False


def download_gait_it(data_dir):
    """Download GAIT-IT dataset info (requires license)."""
    print("\n" + "=" * 60)
    print("GAIT-IT Dataset (Hemiplegic/Pathological Gait)")
    print("=" * 60)
    
    print("  GAIT-IT contains hemiplegic, parkinsonian, and other pathological gaits.")
    print("  Requires license agreement.")
    print("  URL: https://www.it.pt/Groups/Index/72 (search for GAIT-IT)")
    print("  Contact dataset owners for access.")
    
    return False


def check_youtube_videos_for_categories(data_dir):
    """Check GAVD for hemiplegic/stroke videos."""
    print("\n" + "=" * 60)
    print("Checking GAVD for Stroke/Hemiplegic Videos")
    print("=" * 60)
    
    csv_files = list((data_dir / "raw").glob("GAVD_Clinical_Annotations_*.csv"))
    
    if not csv_files:
        print("  [WARNING] No GAVD annotation files found")
        return
    
    import pandas as pd
    
    stroke_videos = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            stroke = df[df['gait_pat'].str.lower() == 'stroke']
            stroke_videos.extend(stroke[['id', 'url']].drop_duplicates().values.tolist())
        except:
            pass
    
    print(f"  Found {len(stroke_videos)} stroke (hemiplegic) videos in GAVD")
    
    if stroke_videos:
        # These are already downloaded to other_abnormal, let's move them
        hemiplegic_dir = data_dir / "raw" / "hemiplegic"
        hemiplegic_dir.mkdir(parents=True, exist_ok=True)
        
        other_dir = data_dir / "raw" / "other_abnormal"
        moved = 0
        for video_id, url in stroke_videos:
            # Check if video exists
            for ext in ['*.mp4', '*.webm', '*.mkv']:
                matches = list(other_dir.glob(f"{video_id}{ext[1:]}"))
                for match in matches:
                    shutil.move(str(match), hemiplegic_dir / match.name)
                    moved += 1
        
        print(f"  [OK] Moved {moved} stroke videos to hemiplegic/")


def main():
    """Main execution."""
    data_dir = project_root / "data"
    
    print("\n" + "=" * 60)
    print("Additional Gait Dataset Downloader")
    print("=" * 60)
    
    # Try to reorganize existing GAVD data first
    check_youtube_videos_for_categories(data_dir)
    
    # Download ROC-HCI Ataxia (most accessible)
    download_roc_hci_ataxia(data_dir)
    
    # Print manual download instructions for others
    download_kaggle_ataxia(data_dir)
    download_mendeley_ataxia(data_dir)
    download_gait_it(data_dir)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    # Count videos in each category
    for category in ['normal', 'hemiplegic', 'parkinsonian', 'ataxic', 'other_abnormal']:
        cat_dir = data_dir / "raw" / category
        if cat_dir.exists():
            videos = list(cat_dir.glob('*.*'))
            print(f"  {category}: {len(videos)} files")
        else:
            print(f"  {category}: 0 files (directory not created)")
    
    print("\n[DONE] Dataset collection complete!")


if __name__ == "__main__":
    main()
