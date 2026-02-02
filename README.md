# Gait Abnormality Detection System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)

A deep learning system that analyzes video input to detect and classify gait abnormalities, providing medical professionals and researchers with automated pattern recognition capabilities for human walking analysis.

## ✨ Features

- 🎥 **Video Processing**: Supports MP4, AVI, MOV formats with quality validation
- 🤸 **Pose Estimation**: MediaPipe-based 33-point body landmark detection
- 🧠 **Deep Learning Models**: Multiple architectures (3D-CNN, LSTM, Hybrid CNN-LSTM)
- 📊 **Gait Analysis**: Comprehensive parameter calculation and abnormality detection
- 🏥 **Clinical Insights**: Detailed reporting with recommendations and risk factors
- 🌐 **Web Interface**: User-friendly Flask-based web application for easy analysis

## 📸 Demo

> **Note**: Add screenshots or GIFs of your web interface here after deployment

<!-- Example:
![Web Interface](docs/images/web-interface.png)
![Analysis Results](docs/images/analysis-results.png)
-->

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 16GB RAM (recommended)
- GPU with 6GB+ VRAM (optional, for training/inference)
- Git installed

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/gait-abnormality-detection.git
   cd gait-abnormality-detection
   ```

2. **Set up the development environment:**
   ```bash
   python setup_env.py
   ```

3. **Activate the virtual environment:**
   
   **Windows:**
   ```cmd
   venv\Scripts\activate
   ```
   
   **Unix/Linux/macOS:**
   ```bash
   source venv/bin/activate
   ```

4. **Set up datasets** (see [DATASET_SETUP.md](DATASET_SETUP.md)):
   ```bash
   python scripts/download_datasets.py
   python scripts/verify_data.py
   ```

### Running the Web Application

```bash
# Navigate to web directory
cd web

# Run the Flask application
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

### Running Analysis via Command Line

```bash
# Analyze a single video
python scripts/analyze_video.py --input path/to/video.mp4 --output results/

# Batch processing
python scripts/batch_process.py --input-dir videos/ --output-dir results/
```

## 📁 Project Structure

```
gait-abnormality-detection/
├── gait_analysis/              # Main package
│   ├── video_processing/       # Video input handling
│   ├── pose_estimation/        # MediaPipe pose detection
│   ├── feature_extraction/     # Feature engineering
│   ├── models/                 # Deep learning model architectures
│   ├── classification/         # Classification logic
│   ├── analysis/               # Gait analysis and reporting
│   ├── training/               # Model training utilities
│   ├── validation/             # Validation and evaluation
│   └── utils/                  # Common utilities
├── web/                        # Web application
│   ├── app.py                  # Flask application
│   ├── static/                 # CSS, JS, images
│   └── templates/              # HTML templates
├── scripts/                    # Utility scripts
│   ├── download_datasets.py    # Dataset download automation
│   ├── verify_data.py          # Data verification
│   └── train_model.py          # Model training script
├── tests/                      # Test suite
├── data/                       # Data directories (not in repo)
│   ├── raw/                    # Raw video files
│   ├── processed/              # Processed datasets
│   └── pose_data/              # Extracted pose keypoints
├── models/                     # Trained model files
├── notebooks/                  # Jupyter notebooks
├── examples/                   # Example videos and outputs
├── DATASET_SETUP.md           # Dataset setup instructions
├── CONTRIBUTING.md            # Contribution guidelines
└── requirements.txt           # Python dependencies
```

## 🔬 Development Approach

This project follows a **hybrid cloud + local development** strategy:

- **Local Development** (RTX 4050): Video processing, pose estimation, prototyping
- **Cloud Training** (Kaggle/Colab): Model training and architecture comparison
- **Local Inference** (RTX 4050): Real-time analysis and demonstration

## 📦 Core Dependencies

- **TensorFlow** 2.13+ - Deep learning framework
- **OpenCV** 4.8+ - Video processing
- **MediaPipe** 0.10+ - Pose estimation
- **Flask** - Web application framework
- **NumPy, Pandas, SciPy** - Scientific computing
- **Matplotlib, Seaborn** - Visualization
- **Pytest, Hypothesis** - Testing

See [requirements.txt](requirements.txt) for complete list.

## 🧪 Testing

The project uses a dual testing approach:
- **Unit Tests**: Specific examples and edge cases
- **Property-Based Tests**: Universal properties with Hypothesis

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=gait_analysis --cov-report=html

# Run specific test file
pytest tests/test_video_processing.py
```

## 📊 Dataset Information

> **Important**: This repository does NOT include the training datasets (~31 GB) due to size constraints.

To set up the datasets:

1. See [DATASET_SETUP.md](DATASET_SETUP.md) for detailed instructions
2. Download datasets from provided cloud storage links
3. Run verification: `python scripts/verify_data.py`

## 🏗️ Model Architecture

The system supports multiple deep learning architectures:

- **3D-CNN**: Spatial-temporal feature extraction from video sequences
- **LSTM**: Temporal pattern recognition in gait sequences
- **Hybrid CNN-LSTM**: Combined spatial and temporal analysis
- **Transfer Learning**: Pre-trained models for improved performance

## 🎯 Use Cases

- **Clinical Assessment**: Automated gait analysis for medical diagnosis
- **Rehabilitation Monitoring**: Track patient progress over time
- **Research**: Large-scale gait pattern analysis
- **Sports Science**: Athletic performance evaluation

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Quick contribution steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MediaPipe team for pose estimation framework
- TensorFlow team for deep learning tools
- Open-source community for various libraries and tools

## 📞 Support

- **Issues**: Report bugs or request features via [GitHub Issues](../../issues)
- **Discussions**: Join conversations in [GitHub Discussions](../../discussions)
- **Documentation**: Check [DATASET_SETUP.md](DATASET_SETUP.md) and [CONTRIBUTING.md](CONTRIBUTING.md)

## 📈 Roadmap

- [ ] Real-time video analysis
- [ ] Mobile application support
- [ ] Additional gait abnormality classifications
- [ ] Integration with medical record systems
- [ ] Multi-person tracking and analysis

---

**Made with ❤️ for advancing gait analysis research and clinical applications**