# 🖼️ Clickbait Image Classifier - ML Pipeline & Cloud Deployment

A comprehensive machine learning pipeline for classifying clickbait images using deep learning, featuring model deployment, monitoring, retraining capabilities, and cloud scalability.

## 📺 Video Demo
**Vide Demo Link:** [Clickbait Image Classifier Demo](https://www.loom.com/share/bba2e806502545d18bb75b392aea608c?sid=bcdab2b5-4fcb-458b-b09b-7e1d3de32d7f)
> *Complete walkthrough of the application, API endpoints, dashboard features, and deployment process*

## 📋 Project Description

This project implements an end-to-end machine learning pipeline for detecting clickbait images using Convolutional Neural Networks (CNNs). The system classifies images as either "fake clickbait" or "legitimate content" and includes comprehensive monitoring, retraining capabilities, and cloud deployment features.

### Key Features:
- **Image Classification**: CNN-based model for clickbait detection with 94.2% accuracy
- **RESTful API**: FastAPI-based prediction service with comprehensive endpoints
- **Interactive Dashboard**: Streamlit UI for monitoring and management
- **Model Retraining**: Automated retraining with new data uploads
- **Load Testing**: Locust-based performance testing and flood simulation
- **Cloud Deployment**: Docker containerization for scalability
- **Real-time Monitoring**: System uptime and performance tracking
- **Data Visualization**: Feature analysis and model interpretation

### Technologies Used:
- **Machine Learning**: TensorFlow/Keras, CNN Architecture
- **Backend**: FastAPI, Python 3.8+
- **Frontend**: Streamlit Dashboard
- **Deployment**: Docker, Cloud Services (AWS/GCP)
- **Testing**: Locust for load testing
- **Data Processing**: NumPy, OpenCV, PIL

## 🚀 Setup Instructions

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/clickbait-image-classifier.git
cd clickbait-image-classifier
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Prepare Training Data
Create the following directory structure and add your images:
```
data/
├── train/
│   ├── clickbait_fake/    # Add 500+ fake clickbait images
│   └── clickbait_real/    # Add 500+ legitimate images
└── test/
    ├── clickbait_fake/    # Add 100+ test fake images
    └── clickbait_real/    # Add 100+ test real images
```

### Step 5: Train Model (Optional - Pre-trained model included)
```bash
cd src
python model.py
```

### Step 6: Start the API Server
```bash
uvicorn src.prediction:app --host 0.0.0.0 --port 8000 --reload
```

### Step 7: Launch Dashboard (New Terminal)
```bash
streamlit run app_hf.py --server.port 8501
```

### Step 8: Access Applications
- **API Documentation**: http://localhost:8000/docs
- **API Info**: http://localhost:8000/docs-info
- **Dashboard**: http://localhost:8501
- **Health Check**: http://localhost:8000/status

### Step 9: Test the System
```bash
# Test single prediction
curl -X POST "http://localhost:8000/predict" -F "file=@sample.jpg"

# Check system status
curl -X GET "http://localhost:8000/system-check"
```

## 📓 Notebook Details

### Jupyter Notebook: `notebook/clickbait_classifier.ipynb`

### Performance Test Results

<img width="1587" height="381" alt="Screenshot 2025-08-03 at 11 03 59" src="https://github.com/user-attachments/assets/850fe15c-5ad6-4ab7-9f73-3e13d1d25961" />


## 🐳 Docker Deployment

### Quick Docker Setup
```bash
# Build image
docker build -t clickbait-classifier .

# Run container
docker run -p 8000:8000 -p 8501:8501 clickbait-classifier

# With volume mounting
docker run -p 8000:8000 -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  clickbait-classifier
```

# 🧪 API Endpoints

### Core Endpoints
- `GET /` - Root endpoint with API status
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /docs-info` - API information and usage examples
- `GET /redoc` - Alternative documentation (ReDoc)
- `GET /status` - Model and system status
- `GET /system-check` - Comprehensive system health check

### Prediction Endpoints
- `POST /predict` - Single image classification
- `POST /predict-batch` - Multiple image classification
- `GET /model-info` - Model performance metrics

### Management Endpoints
- `POST /retrain` - Trigger model retraining
- `GET /debug` - Debug information and request tracking


