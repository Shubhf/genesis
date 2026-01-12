"""
Improved Voice Pipeline
Integrates:
- Enhanced image ingestion (OCR + captioning)
- Embedding-based retrieval
- Persistent memory
- Cross-modal linking
- Smart response routing
"""

import sounddevice as sd
import numpy as np
import queue
import datetime
import os
import requests
import re
from typing import Optional

from faster_whisper import WhisperModel

# Import our improved modules
from event_memory_v2 import Event, MultimodalMemory, ResponseRouter
from image_ingest_v2 import ingest_image


# ============================================================
# Configuration
# ============================================================

class Config:
    # Audio settings
    SAMPLE_RATE = 16000
    CHUNK_SECONDS = 2
    CHANNELS = 1
    AMPLITUDE_THRESHOLD = 0.02
    
    # Transcription
    MIN_TRANSCRIPT_LEN = 4
    WHISPER_MODEL = "small"  # "tiny", "small", "medium", "large-v3"
    
    # Memory
    DB_PATH = "multimodal_memory.db"
    MAX_EVENTS = 100
    TEMPORAL_WINDOW_SECONDS = 10
    
    # LLM
    OLLAMA_URL = "http://127.0.0.1:11434"
    LLM_MODEL = "llama3.1"
    VLM_MODEL = "llava"
    
    # Filtering
    COMMON_FILLERS = {
        "thank you", "thanks", "okay", "ok", "bye", 
        "yes", "no", "um", "uh", "hmm"
    }


# ============================================================
# Intent Detection
# ============================================================

def detect_intent(text: str) -> str:
    """Classify user intent from transcript"""
    t = text.lower()
    
    # Image queries
    if any(k in t for k in ["what is", "what does", "read", "say", "written", "text"]):
        return "ask_image"
    
    if any(k in t for k in ["summarize", "explain", "describe"]):
        return "ask_image"
    
    # Reference to shown content
    if any(k in t for k in ["this", "that", "shown", "above", "here"]):
        return "reference"
    
    # Commands
    if any(k in t for k in ["show", "open", "find", "search"]):
        return "command"
    
    # Questions
    if any(k in t for k in ["what", "why", "how", "when", "where", "who"]):
        return "question"
    
    return "unknown"


def detect_references(text: str) -> list:
    """Detect what modalities the user might be referring to"""
    refs = []
    t = text.lower()
    
    if any(k in t for k in ["image", "photo", "picture", "diagram", "cartoon"]):
        refs.append("image")
    if any(k in t for k in ["doc", "document", "pdf", "file"]):
        refs.append("document")
    if any(k in t for k in ["video", "clip"]):
        refs.append("video")
    if any(k in t for k in ["said", "spoke", "mentioned", "audio"]):
        refs.append("voice")
    
    return refs


def build_voice_semantic(text: str) -> dict:
    """Build semantic dictionary for voice event"""
    return {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "transcript": text,
        "intent": detect_intent(text),
        "references": detect_references(text),
        "word_count": len(text.split()),
        "confidence": min(0.95, 0.6 + 0.03 * len(text.split()))
    }


# ============================================================
# Response Generation
# ============================================================

def call_llm(prompt: str, model: str = None) -> str:
    """Call Ollama LLM"""
    model = model or Config.LLM_MODEL
    
    try:
        response = requests.post(
            f"{Config.OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        
        data = response.json()
        return data.get("response", "").strip()
        
    except requests.exceptions.ConnectionError:
        print("[WARN] Ollama not running")
        return "I'm having trouble connecting to the language model."
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        return "Sorry, I encountered an error."


def build_grounded_prompt(query: str, context: dict) -> str:
    """Build a prompt grounded in retrieved evidence"""
    
    evidence_blocks = []
    
    # Add linked images (shown by user)
    for i, event in enumerate(context.get("linked", [])):
        if "image" in event.modalities:
            img_sem = event.semantics["image"]
            text = img_sem.get("combined_text", img_sem.get("ocr_text", ""))
            if text:
                evidence_blocks.append(f"[SHOWN IMAGE {i+1}]\n{text}")
    
    # Add retrieved events
    for i, event in enumerate(context.get("retrieved", [])):
        if "image" in event.modalities:
            img_sem = event.semantics["image"]
            text = img_sem.get("combined_text", img_sem.get("ocr_text", ""))
            if text:
                evidence_blocks.append(f"[RETRIEVED IMAGE {i+1}]\n{text}")
        
        elif "voice" in event.modalities:
            transcript = event.semantics["voice"].get("transcript", "")
            if transcript:
                evidence_blocks.append(f"[PAST CONVERSATION {i+1}]\n{transcript}")
    
    evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "No relevant evidence found."
    
    prompt = f"""SYSTEM:
You are a helpful assistant with access to the user's multimodal memory.
Answer questions using ONLY the evidence provided below.
If the evidence doesn't contain the answer, say "I don't have that information."
Be concise and direct.

EVIDENCE:
{evidence_text}

USER QUESTION:
{query}

ANSWER:"""
    
    return prompt


def generate_response(voice_event: Event, memory: MultimodalMemory) -> str:
    """
    Generate a response using the smart routing strategy.
    """
    voice_sem = voice_event.semantics["voice"]
    query = voice_sem["transcript"]
    intent = voice_sem["intent"]
    
    # Get context
    context = memory.get_context_for_query(query, voice_event)
    
    # Analyze and route
    rc = ResponseRouter.analyze_context(query, intent, context)
    
    print(f"[ROUTER] Strategy: {rc.strategy}")
    
    # Strategy: Direct read (high confidence OCR)
    if rc.strategy == "DIRECT_READ":
        texts = []
        for event in rc.linked_images[:2]:  # Limit to 2
            ocr = event.semantics["image"].get("ocr_text", "")
            if ocr:
                texts.append(ocr)
        
        if texts:
            return "Here's what I can read from the image:\n\n" + "\n\n".join(texts)
    
    # Strategy: Caption describe (OCR failed but caption available)
    elif rc.strategy == "CAPTION_DESCRIBE":
        descriptions = []
        for event in rc.linked_images[:2]:
            caption = event.semantics["image"].get("caption", "")
            ocr = event.semantics["image"].get("ocr_text", "")
            
            if caption:
                descriptions.append(f"The image shows: {caption}")
            if ocr:
                descriptions.append(f"Text visible: {ocr}")
        
        if descriptions:
            return "\n\n".join(descriptions)
    
    # Strategy: LLM grounded (complex query needs synthesis)
    elif rc.strategy == "LLM_GROUND":
        prompt = build_grounded_prompt(query, context)
        return call_llm(prompt)
    
    # Strategy: Clarify (no context available)
    elif rc.strategy == "CLARIFY":
        # Check if user is referencing something not captured
        if intent == "ask_image" and not rc.linked_images:
            return "I don't see any image right now. Could you show me what you're referring to?"
        return "I'm not sure what you're referring to. Could you give me more context?"
    
    # Strategy: No context
    elif rc.strategy == "NO_CONTEXT":
        return "I don't have any relevant information to answer that question."
    
    # Fallback
    return "I'm not sure how to help with that."


# ============================================================
# Audio Processing
# ============================================================

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[AUDIO] Status: {status}")
    audio_queue.put(indata.copy())


def get_audio_chunk(seconds: float = None) -> np.ndarray:
    """Collect audio for specified duration"""
    seconds = seconds or Config.CHUNK_SECONDS
    frames_required = int(Config.SAMPLE_RATE * seconds)
    audio_frames = []
    collected = 0
    
    while collected < frames_required:
        frame = audio_queue.get()
        audio_frames.append(frame)
        collected += frame.shape[0]
    
    return np.concatenate(audio_frames, axis=0).flatten()


# ============================================================
# Main Application
# ============================================================

class VoicePipeline:
    """
    Main voice pipeline application.
    """
    
    def __init__(self):
        print("[INIT] Starting Voice Pipeline...")
        
        # Initialize memory
        print("[INIT] Loading memory system...")
        self.memory = MultimodalMemory(
            db_path=Config.DB_PATH,
            max_events=Config.MAX_EVENTS,
            temporal_window_seconds=Config.TEMPORAL_WINDOW_SECONDS
        )
        
        # Initialize Whisper
        print("[INIT] Loading Whisper model...")
        self.whisper = WhisperModel(
            Config.WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )
        
        # Initialize audio stream
        print("[INIT] Starting audio stream...")
        self.stream = sd.InputStream(
            samplerate=Config.SAMPLE_RATE,
            channels=Config.CHANNELS,
            dtype="float32",
            blocksize=1024,
            callback=audio_callback
        )
        self.stream.start()
        
        print("[INIT] Ready!")
    
    def add_image(self, image_path: str) -> Event:
        """Add an image to memory"""
        print(f"\n[IMAGE] Processing: {image_path}")
        
        semantic = ingest_image(image_path, use_caption_fallback=True)
        event = Event("image", semantic)
        self.memory.add_event(event)
        
        print(f"[IMAGE] Stored: {event}")
        return event
    
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio using Whisper"""
        segments, _ = self.whisper.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False
        )
        return " ".join(seg.text.strip() for seg in segments)
    
    def process_voice(self, text: str) -> Optional[str]:
        """Process a voice transcript and generate response"""
        
        # Build semantic
        semantic = build_voice_semantic(text)
        
        # Gate: Skip if intent unclear and no references
        if semantic["intent"] == "unknown" and not semantic["references"]:
            print(f"[VOICE] Skipped (unknown intent): {text[:50]}...")
            return None
        
        # Create event and add to memory
        event = Event("voice", semantic)
        self.memory.add_event(event)
        
        print(f"\n[VOICE] {event}")
        print(f"  Transcript: {text}")
        print(f"  Intent: {semantic['intent']}")
        print(f"  Links: {[eid[:8] for eid in event.linked_event_ids]}")
        
        # Generate response
        response = generate_response(event, self.memory)
        
        return response
    
    def run(self):
        """Main loop"""
        print("\n🎙️ Listening... Speak naturally. (Ctrl+C to stop)")
        
        try:
            while True:
                # Get audio chunk
                audio = get_audio_chunk()
                
                # Gate 1: Silence detection
                if np.max(np.abs(audio)) < Config.AMPLITUDE_THRESHOLD:
                    continue
                
                # Transcribe
                text = self.transcribe(audio).strip()
                
                # Gate 2: Quality filter
                if len(text) < Config.MIN_TRANSCRIPT_LEN:
                    continue
                
                if text.lower() in Config.COMMON_FILLERS:
                    continue
                
                # Process and respond
                response = self.process_voice(text)
                
                if response:
                    print(f"\n🤖 RESPONSE:\n{response}")
                
                # Show recent memory
                print("\n[MEMORY] Recent events:")
                for e in self.memory.get_recent_events(3):
                    print(f"  {e}")
        
        except KeyboardInterrupt:
            print("\n[STOP] Shutting down...")
            self.stream.stop()
            self.stream.close()
            
            # Show final stats
            stats = self.memory.get_stats()
            print("\n[STATS] Final memory state:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stream.stop()
        self.stream.close()


# ============================================================
# Entry Point
# ============================================================

def main():
    """Main entry point"""
    pipeline = VoicePipeline()
    
    # Pre-load images (modify paths as needed)
    image_dir = os.path.expanduser("~/Downloads")
    image_files = [
        "amul_1.jpg",
        "amul_4.jpeg",
        "amul_7.jpeg",
        "amul_9.jpg",
        "amul_10.jpg",
    ]
    
    print("\n[LOADING] Adding images to memory...")
    for filename in image_files:
        path = os.path.join(image_dir, filename)
        if os.path.exists(path):
            try:
                pipeline.add_image(path)
            except Exception as e:
                print(f"[ERROR] Failed to load {filename}: {e}")
        else:
            print(f"[SKIP] Not found: {path}")
    
    # Start main loop
    pipeline.run()


if __name__ == "__main__":
    main()
