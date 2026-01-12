"""
Improved Image Ingestion Module
- Hybrid OCR (Tesseract + EasyOCR)
- VLM captioning fallback when OCR fails
- Confidence scoring
"""

from PIL import Image
import pytesseract
import datetime
import os
import requests
import base64
from typing import Optional

# Try to import easyocr (optional but recommended)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    # Initialize once (lazy loading)
    _easyocr_reader = None
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[WARN] EasyOCR not installed. Run: pip install easyocr")


def get_easyocr_reader():
    """Lazy load EasyOCR reader"""
    global _easyocr_reader
    if _easyocr_reader is None and EASYOCR_AVAILABLE:
        print("[INFO] Loading EasyOCR model (first time only)...")
        _easyocr_reader = easyocr.Reader(['en', 'hi'], gpu=False)
    return _easyocr_reader


def extract_text_tesseract(image_path: str) -> dict:
    """Extract text using Tesseract OCR"""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image).strip()
    
    # Basic confidence heuristic
    confidence = calculate_text_confidence(text)
    
    return {
        "text": text,
        "confidence": confidence,
        "engine": "tesseract"
    }


def extract_text_easyocr(image_path: str) -> dict:
    """Extract text using EasyOCR (better for stylized/multilingual)"""
    reader = get_easyocr_reader()
    if reader is None:
        return {"text": "", "confidence": 0, "engine": "easyocr_unavailable"}
    
    results = reader.readtext(image_path)
    
    texts = []
    total_confidence = 0
    
    for bbox, text, conf in results:
        texts.append(text)
        total_confidence += conf
    
    combined_text = " ".join(texts)
    avg_confidence = total_confidence / len(results) if results else 0
    
    return {
        "text": combined_text,
        "confidence": avg_confidence,
        "detections": len(results),
        "engine": "easyocr"
    }


def calculate_text_confidence(text: str) -> float:
    """
    Heuristic confidence score for OCR output quality.
    """
    if not text:
        return 0.0
    
    words = text.split()
    word_count = len(words)
    
    if word_count == 0:
        return 0.0
    
    # Factor 1: Word count (more words = likely better OCR)
    length_score = min(word_count / 10, 1.0)
    
    # Factor 2: Average word length (3-12 chars is reasonable)
    avg_word_len = sum(len(w) for w in words) / word_count
    word_len_score = 1.0 if 3 <= avg_word_len <= 12 else 0.5
    
    # Factor 3: Alphabetic ratio (more letters vs symbols = cleaner)
    alpha_count = sum(c.isalpha() for c in text)
    alpha_ratio = alpha_count / len(text) if text else 0
    
    # Factor 4: Garbage detection (too many special chars = bad OCR)
    special_count = sum(not c.isalnum() and not c.isspace() for c in text)
    special_ratio = special_count / len(text) if text else 0
    garbage_penalty = max(0, 1 - special_ratio * 3)
    
    return (length_score + word_len_score + alpha_ratio + garbage_penalty) / 4


def hybrid_ocr(image_path: str) -> dict:
    """
    Run both OCR engines and pick the best result.
    """
    # Engine 1: Tesseract (fast, good for clean text)
    tess_result = extract_text_tesseract(image_path)
    
    # Engine 2: EasyOCR (better for stylized/multilingual)
    easy_result = extract_text_easyocr(image_path)
    
    # Decision: prefer longer, higher confidence result
    tess_score = len(tess_result["text"]) * tess_result["confidence"]
    easy_score = len(easy_result["text"]) * easy_result["confidence"]
    
    if easy_score > tess_score * 1.1:  # EasyOCR needs to be notably better
        primary = easy_result
    else:
        primary = tess_result
    
    return {
        "text": primary["text"],
        "confidence": primary["confidence"],
        "engine_used": primary["engine"],
        "tesseract_text": tess_result["text"],
        "easyocr_text": easy_result["text"],
        "tesseract_conf": tess_result["confidence"],
        "easyocr_conf": easy_result.get("confidence", 0)
    }


def get_image_caption(image_path: str, ollama_url: str = "http://127.0.0.1:11434") -> str:
    """
    Use VLM (LLaVA via Ollama) to generate image caption.
    Fallback when OCR fails.
    """
    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": "llava",  # or "llava:13b" for better quality
                "prompt": """Describe this image in detail. Include:
1. Any text or words visible (even if stylized or artistic)
2. People or characters shown
3. Objects and their arrangement
4. Colors and visual style
5. The overall context or message

Be thorough but concise.""",
                "images": [image_b64],
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "").strip()
        else:
            print(f"[WARN] Caption API returned {response.status_code}")
            return ""
            
    except requests.exceptions.ConnectionError:
        print("[WARN] Ollama not running. Caption unavailable.")
        return ""
    except Exception as e:
        print(f"[WARN] Caption failed: {e}")
        return ""


def ingest_image(
    image_path: str,
    use_caption_fallback: bool = True,
    ocr_threshold: float = 0.4,
    min_ocr_length: int = 10
) -> dict:
    """
    Full image ingestion pipeline:
    1. Run hybrid OCR
    2. If OCR weak, generate visual caption
    3. Combine all text for retrieval
    
    Args:
        image_path: Path to image file
        use_caption_fallback: Whether to use VLM if OCR fails
        ocr_threshold: Confidence below which to trigger caption
        min_ocr_length: Minimum OCR chars before triggering caption
    
    Returns:
        Semantic dictionary with all extracted information
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Step 1: OCR extraction
    ocr_result = hybrid_ocr(image_path)
    ocr_text = ocr_result["text"]
    ocr_confidence = ocr_result["confidence"]
    
    print(f"[OCR] {os.path.basename(image_path)}")
    print(f"  Engine: {ocr_result['engine_used']}")
    print(f"  Text: {ocr_text[:80]}..." if len(ocr_text) > 80 else f"  Text: {ocr_text}")
    print(f"  Confidence: {ocr_confidence:.2f}")
    
    # Step 2: Caption if OCR is weak
    caption = ""
    if use_caption_fallback:
        if ocr_confidence < ocr_threshold or len(ocr_text) < min_ocr_length:
            print(f"  [CAPTION] OCR weak, generating caption...")
            caption = get_image_caption(image_path)
            if caption:
                print(f"  Caption: {caption[:80]}..." if len(caption) > 80 else f"  Caption: {caption}")
    
    # Step 3: Combine for retrieval
    combined_parts = []
    if ocr_text:
        combined_parts.append(ocr_text)
    if caption:
        combined_parts.append(caption)
    combined_text = "\n".join(combined_parts)
    
    # Build semantic output
    semantic = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "path": image_path,
        "filename": os.path.basename(image_path),
        
        # OCR data
        "ocr_text": ocr_text,
        "ocr_confidence": ocr_confidence,
        "ocr_engine": ocr_result["engine_used"],
        
        # Caption data
        "caption": caption,
        "has_caption": bool(caption),
        
        # Combined for retrieval
        "combined_text": combined_text,
        
        # Quality indicators
        "text_length": len(combined_text),
        "overall_confidence": max(ocr_confidence, 0.7 if caption else 0)
    }
    
    return semantic


# ============================================================
# Convenience functions
# ============================================================

def quick_ingest(image_path: str) -> dict:
    """Simple wrapper for quick testing"""
    return ingest_image(image_path, use_caption_fallback=True)


def batch_ingest(image_paths: list) -> list:
    """Process multiple images"""
    results = []
    for path in image_paths:
        try:
            semantic = ingest_image(path)
            results.append({"path": path, "success": True, "semantic": semantic})
        except Exception as e:
            results.append({"path": path, "success": False, "error": str(e)})
    return results


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            print(f"\n{'='*60}")
            print(f"Processing: {path}")
            print('='*60)
            try:
                result = ingest_image(path)
                print(f"\nResult:")
                print(f"  Combined text length: {result['text_length']}")
                print(f"  Overall confidence: {result['overall_confidence']:.2f}")
                print(f"  Has caption: {result['has_caption']}")
            except Exception as e:
                print(f"Error: {e}")
    else:
        print("Usage: python image_ingest_v2.py <image_path> [image_path2] ...")
