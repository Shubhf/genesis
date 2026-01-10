from PIL import Image
import pytesseract
import datetime
import os

def ingest_image(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    image = Image.open(image_path)

    ocr_text = pytesseract.image_to_string(image).strip()

    semantic = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "path": image_path,
        "ocr_text": ocr_text,
        "confidence": 0.9 if ocr_text else 0.4
    }

    return semantic
