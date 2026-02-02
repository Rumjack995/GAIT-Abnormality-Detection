# Contributing to Gait Abnormality Detection System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/gait-abnormality-detection.git
   cd gait-abnormality-detection
   ```

3. **Set up the development environment** (see below)

## Development Setup

### Prerequisites

- Python 3.8 or higher
- 16GB RAM (recommended)
- GPU with 6GB+ VRAM (optional, for training)
- Git installed

### Environment Setup

1. **Run the automated setup script:**
   ```bash
   python setup_env.py
   ```

2. **Activate the virtual environment:**
   
   **Windows:**
   ```cmd
   venv\Scripts\activate
   ```
   
   **Unix/Linux/macOS:**
   ```bash
   source venv/bin/activate
   ```

3. **Set up datasets** (see [DATASET_SETUP.md](DATASET_SETUP.md)):
   ```bash
   python scripts/download_datasets.py
   python scripts/verify_data.py
   ```

4. **Install development dependencies:**
   ```bash
   pip install -e .
   ```

### Verify Installation

```bash
# Run tests to ensure everything works
pytest tests/

# Start Jupyter Lab for development
jupyter lab
```

## Code Style

### Python Code Style

We follow **PEP 8** with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Organized in groups (standard library, third-party, local)
- **Docstrings**: Google-style docstrings for all public functions/classes

### Example

```python
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from gait_analysis.utils import load_config


def extract_features(video_path: str, config: dict) -> np.ndarray:
    """
    Extract gait features from video file.
    
    Args:
        video_path: Path to input video file
        config: Configuration dictionary with processing parameters
        
    Returns:
        Feature array of shape (num_frames, num_features)
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video format is unsupported
    """
    # Implementation here
    pass
```

### Type Hints

Use type hints for function signatures:

```python
from typing import List, Dict, Optional, Tuple

def process_batch(
    videos: List[str],
    batch_size: int = 32,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Process multiple videos in batches."""
    pass
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the package structure (e.g., `tests/test_video_processing.py`)
- Use both **unit tests** and **property-based tests** (Hypothesis)

### Unit Test Example

```python
import pytest
from gait_analysis.video_processing import validate_video

def test_validate_video_valid_format():
    """Test that valid video formats are accepted."""
    assert validate_video("test.mp4") == True
    assert validate_video("test.avi") == True
    
def test_validate_video_invalid_format():
    """Test that invalid formats are rejected."""
    with pytest.raises(ValueError):
        validate_video("test.txt")
```

### Property-Based Test Example

```python
from hypothesis import given, strategies as st
from gait_analysis.feature_extraction import normalize_features

@given(st.lists(st.floats(min_value=-1000, max_value=1000), min_size=1))
def test_normalize_features_range(features):
    """Normalized features should be in range [0, 1]."""
    normalized = normalize_features(features)
    assert all(0 <= x <= 1 for x in normalized)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=gait_analysis --cov-report=html

# Run specific test file
pytest tests/test_video_processing.py

# Run tests matching a pattern
pytest tests/ -k "test_validate"
```

## Pull Request Process

### Before Submitting

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, descriptive commits:
   ```bash
   git add .
   git commit -m "Add feature: description of what you added"
   ```

3. **Ensure all tests pass:**
   ```bash
   pytest tests/
   ```

4. **Update documentation** if needed (README, docstrings, etc.)

### Submitting the PR

1. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what changed and why
   - Reference to any related issues
   - Screenshots/videos for UI changes

3. **Wait for review** - maintainers will review and provide feedback

### PR Checklist

- [ ] Code follows the project's style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated (if needed)
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts with main branch

## Issue Reporting

### Bug Reports

When reporting bugs, include:

1. **Description**: Clear description of the bug
2. **Steps to reproduce**: Exact steps to trigger the bug
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**:
   - OS and version
   - Python version
   - GPU/CPU specs
   - Relevant package versions
6. **Error messages**: Full error traceback if applicable
7. **Screenshots/videos**: If relevant

### Feature Requests

When requesting features, include:

1. **Use case**: Why is this feature needed?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Other approaches you've considered
4. **Additional context**: Any other relevant information

## Code Review Guidelines

When reviewing code:

- Be respectful and constructive
- Focus on the code, not the person
- Explain the "why" behind suggestions
- Approve when ready, request changes if needed

## Questions?

If you have questions:

1. Check existing [issues](../../issues)
2. Review the [README](README.md) and [documentation](DATASET_SETUP.md)
3. Open a new issue with the "question" label

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
