from ultralytics import YOLO
from app.config import MODEL_PATH, CONFIDENCE_THRESHOLD

_model = None

def load_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model

def is_model_loaded():
    return _model is not None

def predict(image_path: str):
    model = load_model()
    results = model.predict(image_path, conf=CONFIDENCE_THRESHOLD)
    return results[0]  # возвращаем первый результат (для одного изображения)