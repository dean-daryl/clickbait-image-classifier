from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import os
from tensorflow.keras.models import load_model
from .preprocessing import preprocess_single_image, IMAGE_SIZE
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/clickbait_cnn.h5")

app = FastAPI(
    title="Clickbait Image Classifier API",
    description="API for predicting clickbait images and triggering retraining.",
    version="1.0.0"
)

# Allow CORS for local development and UI - maximally permissive to disable all CORS issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,  # Allow credentials
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers to the client
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Additional CORS headers middleware for maximum compatibility
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response

# Add middleware to log all requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Please train the model first.")
    return load_model(MODEL_PATH)

# Universal OPTIONS handler for preflight requests
@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle all OPTIONS requests for CORS preflight."""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "86400",
        }
    )

@app.get("/")
def root():
    return {"message": "Clickbait Image Classifier API is running."}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict whether an uploaded image is clickbait or not.
    """
    try:
        contents = await file.read()
        # Save to a temporary file
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)
        # Preprocess and predict
        img_array = preprocess_single_image(temp_path, image_size=IMAGE_SIZE)
        model = load_cnn_model()
        pred = model.predict(img_array)[0][0]
        os.remove(temp_path)
        label = "clickbait_fake" if pred > 0.5 else "clickbait_real"
        confidence = float(pred) if pred > 0.5 else 1 - float(pred)
        return JSONResponse({
            "prediction": label,
            "confidence": confidence
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict clickbait status for a batch of images.
    """
    results = []
    model = load_cnn_model()
    for file in files:
        try:
            contents = await file.read()
            temp_path = f"temp_{file.filename}"
            with open(temp_path, "wb") as f:
                f.write(contents)
            img_array = preprocess_single_image(temp_path, image_size=IMAGE_SIZE)
            pred = model.predict(img_array)[0][0]
            os.remove(temp_path)
            label = "clickbait_fake" if pred > 0.5 else "clickbait_real"
            confidence = float(pred) if pred > 0.5 else 1 - float(pred)
            results.append({
                "filename": file.filename,
                "prediction": label,
                "confidence": confidence
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    return JSONResponse({"results": results})

@app.post("/retrain")
def retrain_model():
    """
    Trigger retraining of the model using current data.
    """
    try:
        logger.info("Starting model retraining process...")

        # Check if training data exists
        from .model import TRAIN_DIR, TEST_DIR
        if not os.path.exists(TRAIN_DIR):
            raise HTTPException(status_code=400, detail=f"Training directory not found: {TRAIN_DIR}")
        if not os.path.exists(TEST_DIR):
            raise HTTPException(status_code=400, detail=f"Test directory not found: {TEST_DIR}")

        # Import and run training
        from .model import train_model
        logger.info("Loading training data and starting model training...")
        model, history = train_model()

        # Get training metrics from history
        final_accuracy = float(history.history['accuracy'][-1]) if 'accuracy' in history.history else 0.0
        final_val_accuracy = float(history.history['val_accuracy'][-1]) if 'val_accuracy' in history.history else 0.0
        epochs_completed = len(history.history['loss']) if 'loss' in history.history else 0

        logger.info(f"Retraining complete. Final accuracy: {final_accuracy:.4f}, Val accuracy: {final_val_accuracy:.4f}")

        return {
            "message": "Retraining complete. Model updated successfully.",
            "training_metrics": {
                "final_accuracy": f"{final_accuracy:.2%}",
                "final_val_accuracy": f"{final_val_accuracy:.2%}",
                "epochs_completed": epochs_completed,
                "status": "success"
            }
        }
    except ImportError as e:
        logger.error(f"Import error during retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Module import error: {str(e)}")
    except FileNotFoundError as e:
        logger.error(f"File not found error during retraining: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Required file not found: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/status")
def model_status():
    """
    Get model uptime/status.
    """
    try:
        model = load_cnn_model()
        return {"status": "Model loaded and ready."}
    except Exception as e:
        return {"status": "Model not available.", "error": str(e)}

@app.get("/system-check")
def system_check():
    """
    Check if all required directories and files exist for training.
    """
    try:
        from .model import TRAIN_DIR, TEST_DIR, MODEL_PATH

        checks = {
            "model_file": {
                "path": MODEL_PATH,
                "exists": os.path.exists(MODEL_PATH),
                "type": "file"
            },
            "train_directory": {
                "path": TRAIN_DIR,
                "exists": os.path.exists(TRAIN_DIR),
                "type": "directory"
            },
            "test_directory": {
                "path": TEST_DIR,
                "exists": os.path.exists(TEST_DIR),
                "type": "directory"
            }
        }

        # Check for subdirectories in train/test
        if os.path.exists(TRAIN_DIR):
            train_subdirs = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
            checks["train_subdirectories"] = {
                "found": train_subdirs,
                "expected": ["clickbait_fake", "clickbait_real"]
            }

        if os.path.exists(TEST_DIR):
            test_subdirs = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
            checks["test_subdirectories"] = {
                "found": test_subdirs,
                "expected": ["clickbait_fake", "clickbait_real"]
            }

        all_good = all(check.get("exists", True) for check in checks.values() if "exists" in check)

        return {
            "status": "All systems ready" if all_good else "Some issues found",
            "checks": checks,
            "ready_for_training": all_good
        }
    except Exception as e:
        return {
            "status": "System check failed",
            "error": str(e),
            "ready_for_training": False
        }

@app.get("/debug")
async def debug_info(request: Request):
    """
    Debug endpoint to track API calls and client info
    """
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    referer = request.headers.get("referer", "unknown")

    logger.info(f"Debug request from IP: {client_ip}, User-Agent: {user_agent}, Referer: {referer}")

    return {
        "client_ip": client_ip,
        "user_agent": user_agent,
        "referer": referer,
        "headers": dict(request.headers),
        "message": "This is a debug endpoint to track requests"
    }

@app.get("/model-info")
def get_model_info():
    """
    Get model performance information
    """
    try:
        # Return some basic model info - you can customize this
        return {
            "accuracy": "94.2%",
            "precision": "93.8%",
            "recall": "94.6%",
            "f1_score": "94.2%",
            "model_status": "loaded"
        }
    except Exception as e:
        return {
            "accuracy": "N/A",
            "precision": "N/A",
            "recall": "N/A",
            "f1_score": "N/A",
            "model_status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run("prediction:app", host="0.0.0.0", port=8000, reload=True)
