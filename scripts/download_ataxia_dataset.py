"""
Download Ataxia Gait Dataset from Rochester Box.

This downloads the ROC-HCI ataxia dataset which contains 150 video clips
from 89 participants (24 control, 65 diagnosed with Spinocerebellar Ataxia).
"""

import os
import sys
import requests
from pathlib import Path
import zipfile

project_root = Path(__file__).parent.parent
data_dir = project_root / "data"


def download_rochester_ataxia():
    """Download ataxia dataset from Rochester Box."""
    print("=" * 60)
    print("ROC-HCI Ataxia Dataset (Rochester)")
    print("=" * 60)
    
    # Rochester Box shared folder URL
    # Note: Box shared folders may require browser-based download
    box_url = "https://rochester.box.com/v/AtaxiaDataset"
    
    dest_dir = data_dir / "external" / "rochester_ataxia"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Dataset URL: {box_url}")
    print(f"  Destination: {dest_dir}")
    print()
    print("  This dataset requires manual download from Box:")
    print(f"  1. Open: {box_url}")
    print("  2. Click 'Download' button")
    print(f"  3. Extract to: {dest_dir}")
    print("  4. Then run: python scripts/organize_ataxia_data.py")


def download_mendeley_ataxia():
    """Download from Mendeley Data."""
    print("\n" + "=" * 60)
    print("Mendeley Cerebellar Ataxia Dataset")
    print("=" * 60)
    
    # Try direct Mendeley API
    # Dataset: "Dataset for Gait Analysis of Cerebellar Ataxic Patients"
    mendeley_url = "https://data.mendeley.com/api/datasets"
    
    dest_dir = data_dir / "external" / "mendeley_ataxia"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("  Searching Mendeley for cerebellar ataxia gait dataset...")
    
    try:
        # Search for the dataset
        search_url = "https://data.mendeley.com/public-api/datasets?search=cerebellar+ataxia+gait"
        response = requests.get(search_url, timeout=30)
        
        if response.status_code == 200:
            results = response.json()
            print(f"  Found {len(results.get('data', []))} datasets")
            
            for ds in results.get('data', [])[:3]:
                print(f"    - {ds.get('name', 'Unknown')}")
                print(f"      DOI: {ds.get('doi', 'N/A')}")
        else:
            print(f"  API returned status: {response.status_code}")
            
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    print("\n  Manual download:")
    print("  1. Visit: https://data.mendeley.com/datasets")
    print("  2. Search: 'cerebellar ataxia gait mediapipe'")
    print(f"  3. Download and extract to: {dest_dir}")


def create_synthetic_ataxia_from_normal():
    """Create synthetic ataxic gait by perturbing normal gait patterns."""
    print("\n" + "=" * 60)
    print("Creating Synthetic Ataxic Samples")
    print("=" * 60)
    
    import numpy as np
    
    # Load existing synthetic dataset
    npz_path = data_dir / "synthetic_gait_dataset.npz"
    
    if not npz_path.exists():
        print("  [ERROR] Synthetic dataset not found")
        return
    
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    
    print(f"  Loaded: {X.shape[0]} samples")
    
    # Get normal gait samples
    normal_mask = y == 'normal'
    normal_X = X[normal_mask]
    
    print(f"  Normal samples: {len(normal_X)}")
    
    # Create ataxic gait by adding irregular perturbations to normal gait
    # Ataxic gait characteristics:
    # - Irregular stride timing
    # - Wide-based stance
    # - Uncoordinated movements
    
    np.random.seed(42)
    n_ataxic = 300  # Create 300 synthetic ataxic samples
    
    ataxic_X = []
    for i in range(n_ataxic):
        # Select random normal sample
        idx = np.random.randint(len(normal_X))
        sample = normal_X[idx].copy()
        
        # Add irregular perturbations (ataxia characteristics)
        # Random amplitude variations (uncoordinated)
        amplitude_noise = np.random.normal(0, 0.15, sample.shape)
        sample += amplitude_noise
        
        # Random phase shifts (irregular timing)
        for j in range(sample.shape[1]):
            shift = np.random.randint(-5, 6)
            sample[:, j] = np.roll(sample[:, j], shift)
        
        # Add high-frequency tremor component
        t = np.linspace(0, 2*np.pi, sample.shape[0])
        tremor = 0.1 * np.sin(10 * t + np.random.uniform(0, 2*np.pi))
        sample += tremor.reshape(-1, 1)
        
        ataxic_X.append(sample)
    
    ataxic_X = np.array(ataxic_X)
    ataxic_y = np.array(['ataxic'] * n_ataxic)
    
    # Similarly create hemiplegic (asymmetric) patterns
    print("\n  Creating synthetic hemiplegic samples...")
    
    hemiplegic_X = []
    n_hemiplegic = 300
    
    for i in range(n_hemiplegic):
        idx = np.random.randint(len(normal_X))
        sample = normal_X[idx].copy()
        
        # Hemiplegic gait: asymmetric movement, one side weaker
        # Reduce amplitude on one "side" (assume features 0-7 are left, 8-14 are right)
        weakness_factor = np.random.uniform(0.3, 0.6)
        sample[:, 8:] *= weakness_factor
        
        # Add circumduction pattern (swing leg movement)
        t = np.linspace(0, 4*np.pi, sample.shape[0])
        circumduction = 0.15 * np.sin(t)
        sample[:, 8:12] += circumduction.reshape(-1, 1)
        
        hemiplegic_X.append(sample)
    
    hemiplegic_X = np.array(hemiplegic_X)
    hemiplegic_y = np.array(['hemiplegic'] * n_hemiplegic)
    
    # Combine with original data
    new_X = np.concatenate([X, ataxic_X, hemiplegic_X], axis=0)
    new_y = np.concatenate([y, ataxic_y, hemiplegic_y])
    
    # Save augmented dataset
    output_path = data_dir / "augmented_gait_dataset.npz"
    np.savez(output_path, X=new_X, y=new_y)
    
    print(f"\n  [OK] Created augmented dataset: {output_path}")
    print(f"  Total samples: {len(new_y)}")
    
    # Print class distribution
    unique, counts = np.unique(new_y, return_counts=True)
    print("\n  Class distribution:")
    for label, count in zip(unique, counts):
        print(f"    {label}: {count}")


def main():
    print("\n" + "=" * 60)
    print("Ataxia Dataset Acquisition")
    print("=" * 60)
    
    # Show manual download instructions
    download_rochester_ataxia()
    download_mendeley_ataxia()
    
    # Create synthetic data as fallback
    create_synthetic_ataxia_from_normal()
    
    print("\n" + "=" * 60)
    print("[DONE]")
    print("=" * 60)


if __name__ == "__main__":
    main()
