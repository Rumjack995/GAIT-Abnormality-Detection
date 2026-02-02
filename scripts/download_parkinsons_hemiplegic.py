"""
Download Parkinson's and Hemiplegic gait videos using YouTube search.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
data_dir = project_root / "data" / "raw"


def search_and_download(query, output_dir, max_results=50):
    """Search YouTube and download videos."""
    import yt_dlp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Searching: '{query}'")
    print(f"  Downloading up to {max_results} videos...")
    
    ydl_opts = {
        'format': 'best[height<=720]',
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'noplaylist': True,
        'max_downloads': max_results,
    }
    
    successful = 0
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Search and download
            search_query = f"ytsearch{max_results}:{query}"
            ydl.download([search_query])
            
        # Count downloaded
        successful = len(list(output_dir.glob("*.*")))
        
    except Exception as e:
        print(f"  Error: {e}")
    
    return successful


def main():
    print("\n" + "=" * 60)
    print("DOWNLOADING PARKINSON'S AND HEMIPLEGIC VIDEOS")
    print("=" * 60)
    
    park_dir = data_dir / "parkinsonian"
    hemi_dir = data_dir / "hemiplegic"
    
    # Initial counts
    park_before = len(list(park_dir.glob("*.*"))) if park_dir.exists() else 0
    hemi_before = len(list(hemi_dir.glob("*.*"))) if hemi_dir.exists() else 0
    
    print(f"\nBefore: Parkinsonian={park_before}, Hemiplegic={hemi_before}")
    
    # Search queries for Parkinson's
    print("\n" + "=" * 60)
    print("PARKINSON'S GAIT VIDEOS")
    print("=" * 60)
    
    park_queries = [
        "parkinson gait walking patient",
        "parkinson shuffling gait",
        "festinating gait parkinson",
        "parkinson freezing of gait",
        "bradykinesia walking parkinson",
    ]
    
    for query in park_queries:
        search_and_download(query, park_dir, max_results=20)
    
    # Search queries for Hemiplegic/Stroke
    print("\n" + "=" * 60)
    print("HEMIPLEGIC/STROKE GAIT VIDEOS")
    print("=" * 60)
    
    hemi_queries = [
        "hemiplegic gait walking stroke",
        "hemiparesis gait video",
        "stroke patient walking",
        "spastic gait stroke",
        "circumduction gait hemiplegic",
    ]
    
    for query in hemi_queries:
        search_and_download(query, hemi_dir, max_results=20)
    
    # Final counts
    park_after = len(list(park_dir.glob("*.*"))) if park_dir.exists() else 0
    hemi_after = len(list(hemi_dir.glob("*.*"))) if hemi_dir.exists() else 0
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Parkinsonian: {park_before} → {park_after} (+{park_after - park_before})")
    print(f"Hemiplegic: {hemi_before} → {hemi_after} (+{hemi_after - hemi_before})")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
