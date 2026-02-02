#!/usr/bin/env python3
"""
Download script for project datasets.

This script helps collaborators download the required datasets
for the gait abnormality detection project.
"""

import os
import sys
import argparse
from pathlib import Path
import urllib.request
from tqdm import tqdm


class DownloadProgress:
    """Progress bar for downloads."""
    
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True)
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


def download_file(url, destination):
    """Download a file with progress bar."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {destination}")
    
    urllib.request.urlretrieve(
        url,
        destination,
        reporthook=DownloadProgress()
    )
    print(f"✓ Download complete: {destination}")


def download_datasets(data_dir="data", source="gdrive"):
    """
    Download datasets from specified source.
    
    Args:
        data_dir: Directory to save datasets
        source: Download source ('gdrive', 'mega', 'custom')
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Dataset URLs - UPDATE THESE WITH YOUR ACTUAL LINKS
    DATASET_URLS = {
        "gdrive": {
            "raw_videos": "https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_ID",
            "processed_data": "https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_ID",
            "pose_data": "https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_ID",
        },
        "mega": {
            "raw_videos": "https://mega.nz/YOUR_MEGA_LINK",
            "processed_data": "https://mega.nz/YOUR_MEGA_LINK",
            "pose_data": "https://mega.nz/YOUR_MEGA_LINK",
        },
        "custom": {
            "raw_videos": "YOUR_CUSTOM_URL",
            "processed_data": "YOUR_CUSTOM_URL",
            "pose_data": "YOUR_CUSTOM_URL",
        }
    }
    
    print("=" * 60)
    print("Dataset Download Script")
    print("=" * 60)
    print(f"\nDownload source: {source}")
    print(f"Target directory: {data_path.absolute()}\n")
    
    if source not in DATASET_URLS:
        print(f"❌ Error: Unknown source '{source}'")
        print(f"Available sources: {', '.join(DATASET_URLS.keys())}")
        return False
    
    urls = DATASET_URLS[source]
    
    # Check if URLs are configured
    if "YOUR_" in str(urls.values()):
        print("⚠️  Warning: Dataset URLs not configured!")
        print("\nPlease update the DATASET_URLS dictionary in this script with actual download links.")
        print("\nAlternatively, download datasets manually:")
        print("  1. Download from your cloud storage")
        print("  2. Extract to the 'data/' directory")
        print("  3. Run: python scripts/verify_data.py")
        return False
    
    # Download each dataset
    try:
        for dataset_name, url in urls.items():
            filename = f"{dataset_name}.zip"
            destination = data_path / filename
            
            print(f"\n📦 Downloading {dataset_name}...")
            download_file(url, str(destination))
            
            # Auto-extract if it's a zip file
            if destination.suffix == '.zip':
                print(f"📂 Extracting {filename}...")
                import zipfile
                with zipfile.ZipFile(destination, 'r') as zip_ref:
                    zip_ref.extractall(data_path)
                print(f"✓ Extracted successfully")
                
                # Optionally remove zip file after extraction
                destination.unlink()
                print(f"🗑️  Removed {filename}")
        
        print("\n" + "=" * 60)
        print("✅ All datasets downloaded successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Verify data: python scripts/verify_data.py")
        print("  2. Run training: python scripts/train_model.py")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        print("\nTroubleshooting:")
        print("  - Check your internet connection")
        print("  - Verify the download URLs are correct")
        print("  - Try downloading manually from the cloud storage")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for gait abnormality detection project"
    )
    parser.add_argument(
        "--source",
        choices=["gdrive", "mega", "custom"],
        default="gdrive",
        help="Dataset source (default: gdrive)"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory to save datasets (default: data)"
    )
    
    args = parser.parse_args()
    
    success = download_datasets(args.data_dir, args.source)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
