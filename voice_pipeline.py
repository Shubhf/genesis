import sounddevice as sd
import numpy as np
import queue
import datetime
import uuid
from PIL import Image
import pytesseract

import requests

from faster_whisper import WhisperModel
import os 
# ======================
# CONFIG
# ======================
SAMPLE_RATE = 16000
CHUNK_SECONDS = 2
CHANNELS = 1
USE_LLM = True
AMPLITUDE_THRESHOLD = 0.02
MIN_TRANSCRIPT_LEN = 4


COMMON_FILLERS = {
    "thank you",
    "thanks",
    "okay",
    "ok",
    "bye",
    "yes",
    "no"
}
import re

def tokenize(text: str):
    return set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))

# ======================
# EVENT + MEMORY
# ======================
class Event:
    def __init__(self, modality: str, semantic: dict):
        self.event_id = str(uuid.uuid4())
        self.timestamp = semantic["timestamp"]
        self.modalities = [modality]
        self.semantics = {modality: semantic}

    def __repr__(self):
        return f"<Event {self.event_id[:8]} | {self.modalities} | {self.timestamp}>"

class EventMemory:
    def __init__(self, max_events=50):
        self.events = []
        self.max_events = max_events

    def add_event(self, event: Event):
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)

    def get_recent_events(self, n=5):
        return self.events[-n:]

    def get_last_event_by_modality(self, modality: str):
        for event in reversed(self.events):
            if modality in event.modalities:
                return event
        return None

memory = EventMemory(max_events=50)
def retrieve_relevant_images(query: str, memory: EventMemory, k=2):
    scored = []

    q = query.lower()
    for event in memory.events:
        if "image" not in event.modalities:
            continue

        ocr = event.semantics["image"].get("ocr_text", "").lower()
        if not ocr:
            continue

        # naive score = keyword overlap
        score = sum(1 for w in q.split() if w in ocr)

        if score > 0:
            scored.append((score, event))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:k]]


def retrieve_evidence(query: str, memory: EventMemory):
    """
    Returns a list of evidence blocks with metadata.
    Naive lexical retrieval.
    """
    query_tokens = tokenize(query)
    evidence = []

    for event in memory.events:
        if "image" not in event.modalities:
            continue

        img_sem = event.semantics["image"]
        ocr_text = img_sem.get("ocr_text", "")
        if not ocr_text:
            continue

        ocr_tokens = tokenize(ocr_text)
        overlap = query_tokens & ocr_tokens
        score = len(overlap)

        if score > 0:
            evidence.append({
                "event_id": event.event_id,
                "modality": "image",
                "score": score,
                "text": ocr_text
            })

    # sort by relevance
    evidence.sort(key=lambda x: x["score"], reverse=True)
    return evidence


# ======================
# AUDIO QUEUE
# ======================
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    audio_queue.put(indata.copy())

# ======================
# AUDIO STREAM
# ======================
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="float32",
    blocksize=1024,
    callback=audio_callback
)

stream.start()
print("🎙️ Listening... Speak naturally.")

# ======================
# WHISPER MODEL
# ======================
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

# ======================
# AUDIO COLLECTION
# ======================
def get_audio_chunk(seconds=CHUNK_SECONDS):
    frames_required = int(SAMPLE_RATE * seconds)
    audio_frames = []
    collected = 0

    while collected < frames_required:
        frame = audio_queue.get()
        audio_frames.append(frame)
        collected += frame.shape[0]

    return np.concatenate(audio_frames, axis=0).flatten()

# ======================
# TRANSCRIPTION
# ======================
def transcribe(audio_np):
    segments, _ = model.transcribe(
        audio_np,
        language="en",
        beam_size=1,
        vad_filter=False
    )
    return " ".join(seg.text.strip() for seg in segments)

# ======================
# INTENT DETECTION
# ======================
INTENT_KEYWORDS = {
    "ask_question": ["what", "why", "how", "explain"],
    "reference": ["this", "that", "earlier", "shown"],
    "command": ["show", "open", "do"]
}

def detect_intent(text):
    t = text.lower()
    for intent, kws in INTENT_KEYWORDS.items():
        if any(k in t for k in kws):
            return intent
    return "unknown"

# ======================
# REFERENCE DETECTION
# ======================
REFERENCE_KEYWORDS = {
    "image": ["image", "photo", "picture", "diagram"],
    "document": ["doc", "document", "pdf", "file"],
    "video": ["video", "clip"]
}

def detect_references(text):
    refs = []
    t = text.lower()
    for ref, kws in REFERENCE_KEYWORDS.items():
        if any(k in t for k in kws):
            refs.append(ref)
    return refs

# ======================
# VOICE SEMANTIC
# ======================
def build_voice_semantic(text):
    return {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "transcript": text,
        "intent": detect_intent(text),
        "references": detect_references(text),
        "confidence": min(0.95, 0.6 + 0.05 * len(text.split()))
    }
# ======================
# IMAGE INGESTION
# ======================
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
def add_image_event(image_path: str):
    semantic = ingest_image(image_path)
    event = Event(modality="image", semantic=semantic)
    memory.add_event(event)

    print("\nIMAGE EVENT STORED")
    print(event)

    print("Recent memory:")
    for e in memory.get_recent_events(3):
        print(" ", e)

def grounded_response_no_llm(event: Event, memory: EventMemory):
    query = event.semantics["voice"]["transcript"]
    evidence = retrieve_evidence(query, memory)

    if not evidence:
        return "I don’t have any relevant information to answer that."

    top = evidence[0]
    return (
        "Based on the relevant image, here is what I found:\n"
        f"{top['text']}"
    )


import requests

def call_llm(prompt: str) -> str:
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()

    if "response" in data:
        return data["response"].strip()

    # fallback for chat-style responses
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"].strip()

    # last resort
    return "I don't know."




    

def build_grounded_prompt(event: Event, memory: EventMemory):
    voice = event.semantics["voice"]
    query = voice["transcript"]

    image_events = retrieve_relevant_images(query, memory, k=3)

    evidence_blocks = []

    for i, img_event in enumerate(image_events):
        img_sem = img_event.semantics["image"]
        ocr = img_sem.get("ocr_text", "").strip()
        path = os.path.basename(img_sem.get("path", "unknown"))

        block = f"[IMAGE {i+1}]\n"
        block += f"Source: {path}\n"

        if ocr:
            block += f"OCR Text:\n{ocr}\n"
        else:
            block += "OCR Text: <no readable text>\n"

        evidence_blocks.append(block)

    evidence_text = (
        "\n\n".join(evidence_blocks)
        if evidence_blocks
        else "No relevant image evidence was retrieved."
    )

    prompt = f"""
SYSTEM:
You are a grounded multimodal assistant.

Rules you MUST follow:
- Use ONLY the evidence provided below.
- Do NOT use outside knowledge.
- If the evidence does not contain the answer, say exactly: "I don't know."
- If multiple images are provided, reason over them carefully.
- Do not guess or infer unstated visual details.

EVIDENCE:
{evidence_text}

USER QUESTION:
{query}

TASK:
Answer the question using only the evidence.
If the answer is not explicitly supported, respond with "I don't know."
"""

    return prompt.strip()




def generate_response(event: Event, memory: EventMemory):
    prompt = build_grounded_prompt(event, memory)
    return call_llm(prompt)



add_image_event("/Users/shubhgarg/Downloads/amul.jpeg")
add_image_event("/Users/shubhgarg/Downloads/amul_1.jpg")
add_image_event("/Users/shubhgarg/Downloads/amul_4.jpeg")
add_image_event("/Users/shubhgarg/Downloads/amul_5.jpeg")
add_image_event("/Users/shubhgarg/Downloads/amul_6.jpeg")
add_image_event("/Users/shubhgarg/Downloads/amul_7.jpeg")
add_image_event("/Users/shubhgarg/Downloads/amul_8.jpeg")
add_image_event("/Users/shubhgarg/Downloads/amul_9.jpeg")
add_image_event("/Users/shubhgarg/Downloads/amul_10.jpg")







# ======================
# MAIN LOOP
# ======================
try:
    while True:
        audio = get_audio_chunk()

        # ---- Gate 1: silence / noise ----
        if np.max(np.abs(audio)) < AMPLITUDE_THRESHOLD:
            continue

        text = transcribe(audio).strip()

        # ---- Gate 2: transcript quality ----
        if len(text) < MIN_TRANSCRIPT_LEN:
            continue

        if text.lower() in COMMON_FILLERS:
            continue

        semantic = build_voice_semantic(text)

        # ---- Gate 3: event emission ----
        if semantic["intent"] == "unknown" and not semantic["references"]:
            continue

        # ---- STORE EVENT ----
        event = Event(modality="voice", semantic=semantic)
        memory.add_event(event)

        print("\nVOICE EVENT STORED")
        print(event)

        response = generate_response(event, memory)
        if response:
            print("\n🤖 RESPONSE")
            print(response)

        print("Recent memory:")
        for e in memory.get_recent_events(3):
            print(" ", e)


except KeyboardInterrupt:
    print("\nStopping...")
    stream.stop()
    stream.close()
