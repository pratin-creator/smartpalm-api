import io
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf

app = FastAPI(title="SmartPalm API", version="2.0.0-tflite")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "smartpalm_mobilenetv2-3.tflite")
IMG_SIZE = 224

# แก้ให้ตรงกับ label ตอน train
CLASS_NAMES = [
    "GANODERMA",
    "healthy",
    "leaf_spot_blight",
    "nutrient_deficiency",
    "pest_damage",
]

# แก้ mapping ให้ตรงกับ frontend
DISEASE_CODE_MAP = {
    "GANODERMA": "GANODERMA",
    "healthy": "healthy",
    "leaf_spot_blight": "leaf_spot_blight",
    "nutrient_deficiency": "nutrient_deficiency",
    "pest_damage": "pest_damage",
}

# โหลด TFLite ครั้งเดียวตอน startup
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.get("/")
def root():
    return {"status": "ok", "model": "mobilenetv2-tflite", "classes": CLASS_NAMES}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        x = preprocess(image_bytes)

        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]["index"])[0]

        idx = int(np.argmax(preds))
        label = CLASS_NAMES[idx]
        confidence = float(preds[idx]) * 100.0

        return {
            "disease_code": DISEASE_CODE_MAP.get(label, "unknown"),
            "label": label,
            "confidence": round(confidence, 2),
            "model_version": "mobilenetv2-tflite-v3",
            "all_scores": {CLASS_NAMES[i]: round(float(preds[i]) * 100, 2) for i in range(len(CLASS_NAMES))},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
