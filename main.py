from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import io
from datetime import datetime

app = FastAPI(title="SmartPalm AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("smartpalm_mobilenetv2.h5")

class_names = [
    "ganoderma",
    "healthy",
    "leaf_spot_blight",
    "nutrient_deficiency",
    "pest_damage",
    "unknown"
]

disease_info = {
    "ganoderma": {
        "name_th": "โรคโคนเน่ากาโนเดอร์มา",
        "severity": "สูง",
        "color": "red",
        "advice": "ควรตรวจสอบโคนต้น หากพบดอกเห็ดหรือโคนต้นผุ ควรแยกพื้นที่และปรึกษาเจ้าหน้าที่เกษตร"
    },
    "healthy": {
        "name_th": "ปาล์มปกติ",
        "severity": "ต่ำ",
        "color": "green",
        "advice": "ต้นปาล์มมีลักษณะปกติ ควรดูแลตามรอบปกติและติดตามอาการอย่างสม่ำเสมอ"
    },
    "leaf_spot_blight": {
        "name_th": "โรคใบจุด/ใบไหม้",
        "severity": "ปานกลาง",
        "color": "orange",
        "advice": "ควรตัดใบที่เป็นโรครุนแรงออก ลดความชื้นสะสม และเฝ้าระวังการลุกลามของแผล"
    },
    "nutrient_deficiency": {
        "name_th": "ขาดธาตุอาหาร",
        "severity": "ปานกลาง",
        "color": "yellow",
        "advice": "ควรตรวจสภาพดินและพิจารณาเสริมธาตุอาหาร เช่น แมกนีเซียม โพแทสเซียม หรือไนโตรเจน"
    },
    "pest_damage": {
        "name_th": "ความเสียหายจากแมลง",
        "severity": "ปานกลาง",
        "color": "orange",
        "advice": "ควรตรวจใต้ใบและซอกกาบใบ หากพบเพลี้ย หนอน หรือคราบน้ำหวาน ควรจัดการศัตรูพืชอย่างเหมาะสม"
    },
    "unknown": {
        "name_th": "ไม่สามารถระบุได้",
        "severity": "ไม่ทราบ",
        "color": "gray",
        "advice": "ภาพอาจไม่ชัดหรือไม่ตรงกับกลุ่มโรคที่ระบบรู้จัก ควรถ่ายภาพใหม่ให้เห็นอาการชัดเจนขึ้น"
    }
}

@app.get("/")
def home():
    return {
        "message": "SmartPalm AI API is running",
        "version": "1.1"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    img = load_img(io.BytesIO(image_bytes), target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0]

    class_id = int(np.argmax(pred))
    confidence = float(np.max(pred))
    disease_code = class_names[class_id]

    top_indices = np.argsort(pred)[::-1][:3]
    top3_predictions = []

    for i in top_indices:
        code = class_names[int(i)]
        top3_predictions.append({
            "disease_code": code,
            "disease_name": disease_info[code]["name_th"],
            "confidence": round(float(pred[i]), 4),
            "confidence_percent": round(float(pred[i]) * 100, 2)
        })

    info = disease_info[disease_code]

    return {
        "success": True,
        "filename": file.filename,
        "disease_code": disease_code,
        "disease_name": info["name_th"],
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 2),
        "severity": info["severity"],
        "color": info["color"],
        "advice": info["advice"],
        "top3_predictions": top3_predictions,
        "captured_at": datetime.now().isoformat()
    }