from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import os
from tensorflow.keras.models import load_model
from .preprocessing import preprocess_single_image, IMAGE_SIZE
from typing import List

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/clickbait_cnn.h5")

app = FastAPI(
    title="Clickbait Image Classifier API",
    description="API for predicting clickbait images and triggering retraining.",
    version="1.0.0"
)

# Allow CORS for local development and UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Please train the model first.")
    return load_model(MODEL_PATH)

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
        from model import train_model
        model, history = train_model()
        return {"message": "Retraining complete. Model updated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    uvicorn.run("prediction:app", host="0.0.0.0", port=8000, reload=True)
