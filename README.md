# 🫁 BWRP - Pneumonia Detection AI

**An AI-powered pneumonia detection system using chest X-ray analysis**

[![SDG Goal 3](https://img.shields.io/badge/SDG%20Goal-3%20Good%20Health%20and%20Well--being-blue)](https://sdgs.un.org/goals/goal3)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange)](https://tensorflow.org)
[![React](https://img.shields.io/badge/React-18-61dafb)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688)](https://fastapi.tiangolo.com)

## 🌟 Overview

BWRP (Biomedical Workflow Recognition Platform) is an advanced AI system designed to detect pneumonia from chest X-ray images. This project supports **UN Sustainable Development Goal 3: Good Health and Well-being** by providing accessible medical AI tools for healthcare professionals and researchers.

### ✨ Key Features

- 🤖 **Deep Learning Model**: Custom CNN and transfer learning architectures
- 🔬 **High Accuracy**: Trained on the comprehensive Kaggle pneumonia dataset
- 🌐 **Modern Web Interface**: React frontend with intuitive drag-and-drop functionality
- ⚡ **Fast API**: RESTful backend with real-time predictions
- 🛠️ **Modern Tech Stack**: Rye for Python management, Bun.js for React development
- 📊 **Dataset Integration**: Seamless KaggleHub integration for data access
- 🔒 **Privacy-First**: Local processing, no data storage

## 🏗️ Architecture

```
BWRP/
├── 🐍 Python Backend (FastAPI + TensorFlow)
│   ├── AI Model Training & Inference
│   ├── REST API Endpoints
│   └── Dataset Management
├── ⚛️  React Frontend (TypeScript)
│   ├── Image Upload Interface
│   ├── Real-time Predictions
│   └── Results Visualization
└── 📊 Data Pipeline (KaggleHub)
    ├── Automated Dataset Download
    ├── Preprocessing & Augmentation
    └── Model Training Pipeline
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** with [Rye](https://rye-up.com/) installed
- **Node.js 18+** with [Bun](https://bun.sh/) installed
- **Kaggle API** credentials (optional, for dataset download)

### 1. Clone and Setup Backend

```bash
# Clone the repository
git clone <repository-url>
cd bwrp

# Install Python dependencies with Rye
rye sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### 2. Setup Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install React dependencies
npm install
# or use bun for faster installation
bun install
```

### 3. Run the Application

**Terminal 1 - Backend Server:**

```bash
# From project root
rye run python -m bwrp.api
# or
uvicorn bwrp.api:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 - Frontend Development Server:**

```bash
# From frontend directory
npm start
# or
bun start
```

**Access the application:**

- 🌐 **Frontend**: http://localhost:3000
- 📚 **API Docs**: http://localhost:8000/docs
- 🔧 **API Admin**: http://localhost:8000/redoc

## 🛠️ CLI Usage

BWRP includes a comprehensive command-line interface:

```bash
# Install CLI dependencies
rye add click

# Show available commands
python -m bwrp.cli --help

# Download and explore dataset
python -m bwrp.cli dataset

# Train the AI model
python -m bwrp.cli train --architecture custom_cnn --epochs 20

# Make predictions
python -m bwrp.cli predict path/to/xray.jpg

# Start web server
python -m bwrp.cli serve --host 0.0.0.0 --port 8000

# Run complete demo
python -m bwrp.cli demo
```

## 🧠 AI Model Details

### Dataset

- **Source**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes**: Normal vs Pneumonia
- **Images**: 5,856 X-ray images
- **Split**: Train/Validation/Test

### Model Architectures

#### 1. Custom CNN

```python
# Deep convolutional neural network
- 4 Convolutional blocks with BatchNorm & Dropout
- Global Average Pooling
- Dense layers with regularization
- Binary classification output
```

#### 2. Transfer Learning

```python
# EfficientNetB0 pre-trained model
- Frozen base layers
- Custom classification head
- Fine-tuning capabilities
```

### Performance Metrics

- **Accuracy**: Binary classification accuracy
- **Precision**: Pneumonia detection precision
- **Recall**: Pneumonia detection sensitivity
- **F1-Score**: Harmonic mean of precision and recall

## 📊 API Endpoints

### Core Endpoints

| Method | Endpoint        | Description                             |
| ------ | --------------- | --------------------------------------- |
| `GET`  | `/`             | API information and available endpoints |
| `GET`  | `/health`       | Health check and model status           |
| `POST` | `/predict`      | Upload image for pneumonia prediction   |
| `GET`  | `/model/info`   | Model architecture and details          |
| `GET`  | `/dataset/info` | Dataset information and statistics      |

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Predict pneumonia from image
curl -X POST "http://localhost:8000/predict"
     -H "accept: application/json"
     -H "Content-Type: multipart/form-data"
     -F "file=@chest_xray.jpg"
```

## 🧪 Development Workflow

### 1. Dataset Management

```bash
# Download dataset only
python -m bwrp.cli dataset --download-only

# Explore dataset structure
python -c "from bwrp.data_loader import get_dataset_info; print(get_dataset_info())"
```

### 2. Model Training

```bash
# Quick training (demo)
python -m bwrp.cli train --epochs 5

# Full training with transfer learning
python -m bwrp.cli train --architecture transfer_learning --epochs 50 --model-dir models
```

### 3. Testing & Validation

```bash
# Test API endpoints
python -m pytest tests/

# Validate model predictions
python -m bwrp.cli predict examples/normal_xray.jpg
python -m bwrp.cli predict examples/pneumonia_xray.jpg
```

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Kaggle API credentials
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Optional: Model configuration
export MODEL_PATH=models/custom_model.h5
export BATCH_SIZE=32
export IMAGE_SIZE=224
```

### Project Configuration (`pyproject.toml`)

```toml
[project]
name = "bwrp"
version = "0.1.0"
description = "Pneumonia Detection AI for SDG Goal 3"
dependencies = [
    "tensorflow>=2.19.0",
    "fastapi>=0.116.0",
    "uvicorn>=0.30.0",
    "kagglehub>=0.3.12",
    # ... see full list in pyproject.toml
]
```

## 📈 Performance & Monitoring

### Model Metrics

- Training accuracy and loss tracking
- Validation performance monitoring
- Test set evaluation
- Confusion matrix analysis

### API Monitoring

- Request/response logging
- Performance metrics
- Error tracking
- Health checks

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install development dependencies
rye add --dev pytest black isort mypy

# Run code formatting
black src/
isort src/

# Run type checking
mypy src/

# Run tests
pytest
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Paul Mooney (Kaggle) for the chest X-ray pneumonia dataset
- **SDG Goals**: United Nations Sustainable Development Goals
- **Open Source**: TensorFlow, FastAPI, React, and the entire open-source community
- **Medical Community**: Healthcare professionals fighting pneumonia worldwide

## 📞 Support

- 📧 **Email**: support@bwrp-ai.com
- 💬 **Issues**: [GitHub Issues](https://github.com/your-username/bwrp/issues)
- 📖 **Documentation**: [Project Wiki](https://github.com/your-username/bwrp/wiki)
- 🌐 **Website**: [BWRP AI](https://bwrp-ai.com)

---

**Made with ❤️ for SDG Goal 3: Good Health and Well-being**

_Disclaimer: This AI system is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions._
