import os

# Папка, где находится этот файл (app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

CONFIDENCE_THRESHOLD = 0.25
PIXELS_PER_CM = 93.8