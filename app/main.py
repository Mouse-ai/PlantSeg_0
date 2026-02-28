from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import numpy as np
import cv2
from app import model, utils
from app.config import PIXELS_PER_CM, MODEL_PATH  # импортируем MODEL_PATH
allow_origins=[
    "http://localhost:5173",                 # локальная разработка
    "http://192.168.0.104:5173",              # если открываете по IP
    "https://ваш-проект.vercel.app"           # ваш домен на Vercel
]
app = FastAPI(title="Plant Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Проверяем наличие файла модели по пути из config.py
if not os.path.exists(MODEL_PATH):
    print(f"⚠️  ВНИМАНИЕ: Файл модели не найден по пути {MODEL_PATH}")
    print("Пожалуйста, проверьте правильность пути в config.py")
else:
    try:
        model.load_model()
        print("✅ Модель успешно загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...), scale: float = None):
    if not model.is_model_loaded():
        raise HTTPException(status_code=503, detail="Модель не загружена. Проверьте наличие файла best.pt")

    file_ext = os.path.splitext(file.filename)[1]
    file_name = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, file_name)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = model.predict(file_path)
        boxes = results.boxes
        masks = results.masks

        if masks is None:
            return JSONResponse(content={"predictions": []})

        pixels_per_cm = scale if scale is not None else PIXELS_PER_CM
        predictions = []

        for box, mask_data in zip(boxes, masks.data):
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]
            if class_name == 'steb':
                class_name = 'stem'

            confidence = float(box.conf[0])
            mask = mask_data.cpu().numpy() > 0.5
            measurements = utils.measure_mask(mask, pixels_per_cm)
            polygon = utils.mask_to_polygon(mask)

            predictions.append({
                "class": class_name,
                "confidence": round(confidence, 3),
                "polygon": polygon,
                "area_cm2": measurements["area_cm2"],
                "length_cm": measurements["length_cm"] if class_name in ["root", "stem"] else None
            })

        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/")
async def root():
    return {"message": "Plant Segmentation API is running. Use /predict to upload an image."}