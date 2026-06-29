from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import io
import time
import logging
from datetime import datetime
from typing import Optional

# =========================
# Basic Config
# =========================

MODEL_PATH = "smartpalm_mobilenetv2.h5"
MODEL_NAME = "MobileNetV2"
MODEL_VERSION = "v1.2"
API_VERSION = "1.2"
IMAGE_SIZE = (224, 224)
MAX_FILE_SIZE_MB = 15
UNKNOWN_THRESHOLD = 0.65

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmartPalmAI")

app = FastAPI(
    title="SmartPalm AI API",
    description="FastAPI backend for SmartPalm AI oil palm disease diagnosis using MobileNetV2.",
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loading model
model = None


def get_model():
    global model
    
    if model is None:
        logger.info("Loading AI model...")
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")

    return model


class_names = [
    "ganoderma",
    "healthy",
    "leaf_spot_blight",
    "nutrient_deficiency",
    "pest_damage",
    "unknown",
]


disease_info = {
    "ganoderma": {
        "name_th": "โรคโคนเน่ากาโนเดอร์มา",
        "severity": "สูง",
        "color": "red",
        "advice": "ควรตรวจสอบโคนต้น หากพบดอกเห็ดหรือโคนต้นผุ ควรแยกพื้นที่และปรึกษาเจ้าหน้าที่เกษตร",
    },
    "healthy": {
        "name_th": "ปาล์มปกติ",
        "severity": "ต่ำ",
        "color": "green",
        "advice": "ต้นปาล์มมีลักษณะปกติ ควรดูแลตามรอบปกติและติดตามอาการอย่างสม่ำเสมอ",
    },
    "leaf_spot_blight": {
        "name_th": "โรคใบจุด/ใบไหม้",
        "severity": "ปานกลาง",
        "color": "orange",
        "advice": "ควรตัดใบที่เป็นโรครุนแรงออก ลดความชื้นสะสม และเฝ้าระวังการลุกลามของแผล",
    },
    "nutrient_deficiency": {
        "name_th": "ขาดธาตุอาหาร",
        "severity": "ปานกลาง",
        "color": "yellow",
        "advice": "ควรตรวจสภาพดินและพิจารณาเสริมธาตุอาหาร เช่น แมกนีเซียม โพแทสเซียม หรือไนโตรเจน",
    },
    "pest_damage": {
        "name_th": "ความเสียหายจากแมลง",
        "severity": "ปานกลาง",
        "color": "orange",
        "advice": "ควรตรวจใต้ใบและซอกกาบใบ หากพบเพลี้ย หนอน หรือคราบน้ำหวาน ควรจัดการศัตรูพืชอย่างเหมาะสม",
    },
    "unknown": {
        "name_th": "ไม่สามารถระบุได้",
        "severity": "ไม่ทราบ",
        "color": "gray",
        "advice": "ระบบไม่สามารถระบุโรคได้อย่างมั่นใจ อาจเกิดจากภาพไม่ชัด ถ่ายผิดตำแหน่ง หรืออาการไม่ตรงกับฐานข้อมูล ควรถ่ายภาพใหม่หลายมุม หรือส่งให้ผู้เชี่ยวชาญตรวจสอบ",
    },
}


def confidence_tier(confidence: float) -> str:
    if confidence >= 0.95:
        return "excellent"
    if confidence >= 0.80:
        return "high"
    if confidence >= 0.65:
        return "medium"
    return "low"


def validate_file(file: UploadFile, image_bytes: bytes):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_type": "INVALID_FILE_TYPE",
                "message": "รองรับเฉพาะไฟล์ภาพ JPG, PNG หรือ WEBP เท่านั้น",
            },
        )

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "success": False,
                "error_type": "FILE_TOO_LARGE",
                "message": f"ไฟล์มีขนาดใหญ่เกิน {MAX_FILE_SIZE_MB} MB",
            },
        )


def preprocess_image(image_bytes: bytes):
    img = load_img(io.BytesIO(image_bytes), target_size=IMAGE_SIZE)
    original_size = getattr(img, "size", None)

    img_array = img_to_array(img)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, original_size


def build_top3_predictions(pred):
    top_indices = np.argsort(pred)[::-1][:3]
    top3_predictions = []

    for i in top_indices:
        code = class_names[int(i)]
        confidence = float(pred[i])

        top3_predictions.append(
            {
                "disease_code": code,
                "disease_name": disease_info[code]["name_th"],
                "confidence": round(confidence, 4),
                "confidence_percent": round(confidence * 100, 2),
            }
        )

    return top3_predictions


@app.get("/")
def home():
    return {
        "success": True,
        "message": "SmartPalm AI API is running",
        "api_version": API_VERSION,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_v1": "/v1/predict",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "success": True,
        "api_version": API_VERSION,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        "unknown_threshold": UNKNOWN_THRESHOLD,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return await predict_v1(file)


@app.post("/v1/predict")
async def predict_v1(file: UploadFile = File(...)):
    started_at = time.time()

    try:
        image_bytes = await file.read()
        validate_file(file, image_bytes)

        img_array, original_size = preprocess_image(image_bytes)

        ai_model = get_model()
        pred = ai_model.predict(img_array, verbose=0)[0]

        raw_class_id = int(np.argmax(pred))
        raw_confidence = float(np.max(pred))
        raw_disease_code = class_names[raw_class_id]

        if raw_confidence < UNKNOWN_THRESHOLD:
            disease_code = "unknown"
            class_id = class_names.index("unknown")
            confidence = raw_confidence
        else:
            disease_code = raw_disease_code
            class_id = raw_class_id
            confidence = raw_confidence

        top3_predictions = build_top3_predictions(pred)
        info = disease_info[disease_code]

        inference_time_ms = round((time.time() - started_at) * 1000, 2)

        logger.info(
            f"Prediction success | file={file.filename} | disease={disease_code} | confidence={confidence:.4f} | time={inference_time_ms}ms"
        )

        return {
            "success": True,
            "filename": file.filename,
            "disease_code": disease_code,
            "disease_name": info["name_th"],
            "class_id": class_id,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "confidence_tier": confidence_tier(confidence),
            "severity": info["severity"],
            "color": info["color"],
            "advice": info["advice"],
            "top3_predictions": top3_predictions,
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "api_version": API_VERSION,
            "inference_time_ms": inference_time_ms,
            "input_image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
            "original_image_size": original_size,
            "unknown_threshold": UNKNOWN_THRESHOLD,
            "captured_at": datetime.now().isoformat(),
            "raw_prediction": {
                "raw_class_id": raw_class_id,
                "raw_disease_code": raw_disease_code,
                "raw_confidence": round(raw_confidence, 4),
                "probabilities": {
                    class_names[i]: round(float(pred[i]), 4)
                    for i in range(len(class_names))
                },
            },
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_type": "PREDICTION_ERROR",
                "message": "เกิดข้อผิดพลาดระหว่างวิเคราะห์ภาพ กรุณาลองใหม่อีกครั้ง",
                "technical_detail": str(e),
                "timestamp": datetime.now().isoformat(),
            },

        )
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=10000)