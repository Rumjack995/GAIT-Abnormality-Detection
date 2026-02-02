# Development Guide

## Getting Started

### 1. Environment Setup

The development environment has been set up with all required dependencies. To activate it:

**Windows:**
```cmd
venv\Scripts\activate
```

**Unix/Linux/macOS:**
```bash
source venv/bin/activate
```

### 2. Verify Setup

Run the setup verification notebook:
```bash
jupyter lab notebooks/01_setup_verification.ipynb
```

Or run the test suite:
```bash
pytest tests/test_setup.py -v
```

### 3. Project Structure

```
gait-abnormality-detection/
├── gait_analysis/              # Main package
│   ├── video_processing/       # Video input handling
│   ├── pose_estimation/        # MediaPipe pose detection
│   ├── feature_extraction/     # Feature engineering
│   ├── models/                 # Deep learning models
│   ├── analysis/               # Analysis and reporting
│   └── utils/                  # Common utilities
├── tests/                      # Test suite
├── data/                       # Data directories
├── models/                     # Trained model files
├── notebooks/                  # Jupyter notebooks
└── .kiro/specs/               # Project specifications
```

## Development Workflow

### 1. Task-Based Development

Follow the implementation plan in `.kiro/specs/gait-abnormality-detection/tasks.md`:

1. **Video Processing Pipeline** (Task 2)
2. **Pose Estimation System** (Task 3)
3. **Feature Extraction Pipeline** (Task 4)
4. **Model Architectures** (Task 6)
5. **Training System** (Task 7)
6. **Classification System** (Task 8)
7. **Analysis and Reporting** (Task 9)

### 2. Testing Strategy

- **Unit Tests**: Test specific functionality with concrete examples
- **Property-Based Tests**: Test universal properties with Hypothesis
- **Integration Tests**: Test component interactions

Run tests:
```bash
# All tests
pytest

# Specific test file
pytest tests/test_setup.py

# With coverage
pytest --cov=gait_analysis
```

### 3. Configuration Management

The system uses YAML configuration in `config.yaml`. Access configuration:

```python
from gait_analysis.utils.config import get_config

config = get_config()
batch_size = config.get('training.batch_size')
video_formats = config.get('video.supported_formats')
```

### 4. Hardware Optimization

**Local Development (RTX 4050):**
- Batch size: 2-4 (optimized for 6GB VRAM)
- Mixed precision training enabled
- Memory growth enabled for GPU

**Cloud Training (Kaggle/Colab):**
- Larger batch sizes (8-16)
- Full model training
- Architecture comparison

## Key Dependencies

### Core ML Stack
- **TensorFlow 2.20+**: Deep learning framework
- **OpenCV 4.13+**: Computer vision operations
- **MediaPipe 0.10+**: Pose estimation
- **NumPy/Pandas**: Data processing

### Development Tools
- **Jupyter Lab**: Interactive development
- **pytest**: Testing framework
- **Hypothesis**: Property-based testing

## Common Tasks

### Add New Module
1. Create module directory in `gait_analysis/`
2. Add `__init__.py` file
3. Implement functionality
4. Add tests in `tests/`
5. Update configuration if needed

### Add New Model Architecture
1. Create model class in `gait_analysis/models/`
2. Implement required interface methods
3. Add configuration parameters
4. Create unit tests
5. Add to architecture comparison

### Add New Feature Extractor
1. Implement in `gait_analysis/feature_extraction/`
2. Follow data structure interfaces
3. Add configuration parameters
4. Create property-based tests

## Debugging Tips

### GPU Issues
```python
# Check GPU availability
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# Enable memory growth
gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)
```

### Memory Issues
- Reduce batch size in `config.yaml`
- Enable mixed precision training
- Use gradient accumulation for larger effective batch sizes

### Import Issues
- Ensure virtual environment is activated
- Install package in development mode: `pip install -e .`
- Check PYTHONPATH includes project root

## Next Steps

1. **Implement Video Processing** (Task 2): Start with `VideoProcessor` class
2. **Set up Pose Estimation** (Task 3): Integrate MediaPipe
3. **Create Feature Extraction** (Task 4): Build spatiotemporal features
4. **Develop Models** (Task 6): Implement 3D-CNN, LSTM, and Hybrid architectures

Each task includes specific requirements and property-based tests to ensure correctness.