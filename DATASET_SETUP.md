# Dataset Setup Guide

This project uses large datasets (~31 GB) that are not included in the repository due to size constraints.

## Dataset Structure

The project expects the following data directory structure:

```
data/
├── raw/              # Raw video files (MP4, AVI, MOV)
├── processed/        # Processed features (PKL, H5, NPY)
├── pose_data/        # Extracted pose keypoints
└── external/         # External datasets
```

## Setup Options

### Option 1: Download from Cloud Storage (Recommended)

Download the datasets from one of these sources:

**Google Drive:**
```
[Add your Google Drive link here]
```

**OneDrive/Dropbox/Other:**
```
[Add alternative download link here]
```

After downloading:
```bash
# Extract to project root
unzip datasets.zip -d data/
```

### Option 2: Use Your Own Data

If you want to train with your own gait videos:

1. Place raw video files in `data/raw/`
2. Run the preprocessing pipeline:
```bash
python scripts/preprocess_data.py
```

### Option 3: Download Public Datasets

Use the provided download script to fetch public gait datasets:

```bash
python scripts/download_datasets.py
```

## Verification

After setup, verify your data is correctly placed:

```bash
python scripts/verify_data.py
```

Expected output:
```
✓ data/raw/: 500 video files found
✓ data/processed/: Feature files generated
✓ data/pose_data/: Pose keypoints extracted
```

## Dataset Information

- **Total Size:** ~31 GB
- **Video Format:** MP4, AVI, MOV
- **Processed Format:** PKL, H5, NPY
- **Required for:** Training, evaluation, and testing

## Sharing Your Trained Models

If you want to share only your trained models without datasets:

```bash
# Models are much smaller (~100 MB)
# Already included if not excluded by .gitignore
models/
├── trained_model.h5
└── classifier.pkl
```

## Questions?

See [README.md](README.md) for more information or file an issue.
