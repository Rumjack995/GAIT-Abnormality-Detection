# GAVD Dataset Setup - Quick Start Guide

## 📋 Overview

This guide will help you download, preprocess, and train models on the GAVD (Gait Abnormality in Video Dataset).

**Estimated Time**: 1-2 days (depending on dataset size and GPU)

---

## 🚀 Step-by-Step Instructions

### Step 1: Download GAVD Dataset (1-2 hours)

```bash
# Activate virtual environment
venv\Scripts\activate

# Run download script
python scripts\download_gavd.py
```

**What this does**:
- Downloads GAVD repository from GitHub
- Organizes videos into categories (normal, hemiplegic, parkinsonian, ataxic, other_abnormal)
- Creates data directory structure

**Expected output**:
```
data/raw/
├── normal/
├── hemiplegic/
├── parkinsonian/
├── ataxic/
└── other_abnormal/
```

**Note**: If the repository contains video URLs instead of actual videos, you'll need to download them manually. The script will guide you.

---

### Step 2: Preprocess Dataset (2-4 hours)

```bash
# Run preprocessing script
python scripts\preprocess_dataset.py
```

**What this does**:
- Validates all videos (format, resolution, duration)
- Extracts frames at 30 FPS
- Applies quality enhancement
- Runs MediaPipe pose estimation
- Creates train/val/test splits (70%/15%/15%)
- Saves processed data as `.pkl` files

**Expected output**:
```
data/processed/
├── normal/
│   ├── video_001.pkl
│   ├── video_002.pkl
│   └── ...
├── hemiplegic/
├── parkinsonian/
├── ataxic/
├── other_abnormal/
└── dataset_splits.json
```

**Progress tracking**: You'll see progress bars for each category during processing.

---

### Step 3: Train Models (1-2 days)

```bash
# Run training script
python scripts\train_models.py
```

**What this does**:
- Trains 3 architectures: LSTM, 3D-CNN, Hybrid CNN-LSTM
- Uses data augmentation
- Saves best models
- Compares performance
- Generates training history

**Training order** (fastest to slowest):
1. LSTM (~2-4 hours)
2. 3D-CNN (~4-8 hours)
3. Hybrid (~6-12 hours)

**Expected output**:
```
models/
├── lstm_model.h5
├── lstm_history.json
├── 3dcnn_model.h5
├── 3dcnn_history.json
├── hybrid_model.h5
├── hybrid_history.json
└── model_comparison.json
```

---

## 💡 Tips & Troubleshooting

### GPU Optimization
- **RTX 4050 (6GB VRAM)**: Default batch size of 8 should work
- **Less VRAM**: Reduce batch size in `train_models.py` (line 186)
- **No GPU**: Training will use CPU (much slower, 10-20x)

### Common Issues

#### 1. Git Not Installed
```
❌ ERROR: Git is not installed!
```
**Solution**: Download from https://git-scm.com/download/win

#### 2. No Videos Found
```
⚠️ No videos found in repository
```
**Solution**: GAVD likely provides URLs. Check the repository README for download instructions.

#### 3. Out of Memory During Training
```
ResourceExhaustedError: OOM when allocating tensor
```
**Solution**: Reduce batch size in `train_models.py`:
```python
batch_size=4,  # Change from 8 to 4
```

#### 4. MediaPipe Errors
```
Error: MediaPipe pose detection failed
```
**Solution**: Some videos may have poor quality. These will be skipped automatically.

---

## 📊 Monitoring Progress

### During Preprocessing
- Watch for validation errors
- Check `data/raw/invalid_videos.json` for problematic videos
- Typical success rate: 90-95%

### During Training
- Monitor validation accuracy
- Look for overfitting (train acc >> val acc)
- Best models are saved automatically

**Expected accuracy** (depends on dataset quality):
- LSTM: 70-85%
- 3D-CNN: 75-90%
- Hybrid: 80-95%

---

## 🎯 After Training

### Test Your Model

```python
from gait_analysis.classification import GaitClassifier

# Load best model
classifier = GaitClassifier(architecture_type='hybrid')
classifier.load_model('models/hybrid_model.h5')

# Analyze a new video
from gait_analysis.video_processing import VideoProcessor
from gait_analysis.pose_estimation import PoseEstimator

processor = VideoProcessor()
estimator = PoseEstimator()

# Process video
frames = processor.extract_frames('path/to/video.mp4')
poses = estimator.extract_poses(frames)

# Classify
result = classifier.predict(poses)
print(f"Abnormality: {result.abnormality_type}")
print(f"Confidence: {result.confidence:.2f}")
```

---

## 📁 Directory Structure After Completion

```
major project v1/
├── data/
│   ├── raw/                    # Original videos
│   └── processed/              # Processed .pkl files
├── models/                     # Trained models
│   ├── lstm_model.h5          ✅
│   ├── 3dcnn_model.h5         ✅
│   └── hybrid_model.h5        ✅
├── scripts/
│   ├── download_gavd.py       # Step 1
│   ├── preprocess_dataset.py  # Step 2
│   └── train_models.py        # Step 3
└── temp_download/             # Temporary (can delete after)
```

---

## ⏱️ Timeline Summary

| Step | Time | Can Run Overnight? |
|------|------|-------------------|
| Download | 1-2 hours | ✅ Yes |
| Preprocess | 2-4 hours | ✅ Yes |
| Train LSTM | 2-4 hours | ✅ Yes |
| Train 3D-CNN | 4-8 hours | ✅ Yes |
| Train Hybrid | 6-12 hours | ✅ Yes |

**Total**: 15-30 hours (mostly automated)

---

## 🆘 Need Help?

1. Check error messages in terminal
2. Review log files in `data/raw/invalid_videos.json`
3. Verify GPU is detected: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

---

## ✅ Success Checklist

- [ ] GAVD repository downloaded
- [ ] Videos organized in `data/raw/`
- [ ] Preprocessing completed without major errors
- [ ] Dataset splits created (`dataset_splits.json`)
- [ ] At least one model trained successfully
- [ ] Model comparison generated
- [ ] Can run inference on new videos

**Once complete, your project will be fully functional! 🎉**
