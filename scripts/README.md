# Scripts Directory

This directory contains utility scripts for dataset preparation and model training.

## Available Scripts

### 1. `download_gavd.py`
Downloads and organizes the GAVD (Gait Abnormality in Video Dataset).

**Usage**:
```bash
python scripts/download_gavd.py
```

**What it does**:
- Clones GAVD repository from GitHub
- Organizes videos into category directories
- Creates data structure for training

---

### 2. `preprocess_dataset.py`
Preprocesses videos for training.

**Usage**:
```bash
python scripts/preprocess_dataset.py
```

**What it does**:
- Validates all videos
- Extracts frames at 30 FPS
- Runs pose estimation with MediaPipe
- Creates train/val/test splits
- Saves processed data as `.pkl` files

**Requirements**: Videos must be in `data/raw/` directory

---

### 3. `train_models.py`
Trains all model architectures.

**Usage**:
```bash
python scripts/train_models.py
```

**What it does**:
- Trains LSTM, 3D-CNN, and Hybrid models
- Uses data augmentation
- Saves best models to `models/` directory
- Generates performance comparison

**Requirements**: Preprocessed data must exist in `data/processed/`

---

## Quick Start Workflow

```bash
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Download dataset
python scripts\download_gavd.py

# 3. Preprocess videos
python scripts\preprocess_dataset.py

# 4. Train models
python scripts\train_models.py
```

See `GAVD_SETUP_GUIDE.md` for detailed instructions.
