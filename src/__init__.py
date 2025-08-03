"""
Clickbait Image Classifier - Source Package

This package contains the core modules for the clickbait image classification system:
- model.py: CNN model architecture and training utilities
- prediction.py: FastAPI application for image classification API
- preprocessing.py: Image preprocessing and data generation utilities
"""

from .preprocessing import preprocess_single_image, get_data_generators, IMAGE_SIZE, BATCH_SIZE
from .model import build_mobilenet_model, train_model, evaluate_model, fine_tune_model, predict_single_image

__version__ = "1.0.0"
__author__ = "Clickbait Classifier Team"

__all__ = [
    "preprocess_single_image",
    "get_data_generators",
    "IMAGE_SIZE",
    "BATCH_SIZE",
    "build_mobilenet_model",
    "train_model",
    "evaluate_model",
    "fine_tune_model",
    "predict_single_image"
]
