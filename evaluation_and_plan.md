# Multimodal Memory System: Evaluation & Improvement Plan

## Part 1: Current Implementation Analysis

### What You Have Built

```
┌─────────────────────────────────────────────────────────────────┐
│                    CURRENT ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │  Audio Input │     │ Image Input  │     │  EventMemory │   │
│   │  (sounddevice)│     │  (PIL)       │     │  (in-memory) │   │
│   └──────┬───────┘     └──────┬───────┘     └──────────────┘   │
│          │                    │                     ▲          │
│          ▼                    ▼                     │          │
│   ┌──────────────┐     ┌──────────────┐            │          │
│   │   Whisper    │     │  Tesseract   │            │          │
│   │   (ASR)      │     │   (OCR)      │            │          │
│   └──────┬───────┘     └──────┬───────┘            │          │
│          │                    │                     │          │
│          ▼                    ▼                     │          │
│   ┌──────────────┐     ┌──────────────┐            │          │
│   │ Intent/Ref   │     │  Semantic    │            │          │
│   │  Detection   │     │  Extraction  │            │          │
│   └──────┬───────┘     └──────┬───────┘            │          │
│          │                    │                     │          │
│          └────────┬───────────┘                     │          │
│                   ▼                                 │          │
│            ┌──────────────┐                        │          │
│            │    Event     │────────────────────────┘          │
│            │   Creation   │                                   │
│            └──────┬───────┘                                   │
│                   │                                           │
│                   ▼                                           │
│            ┌──────────────┐     ┌──────────────┐             │
│            │   Lexical    │────▶│   Ollama     │             │
│            │  Retrieval   │     │   (LLM)      │             │
│            └──────────────┘     └──────────────┘             │
│                                                               │
└─────────────────────────────────────────────────────────────────┘
```

### Strengths ✓

| Component | What's Good |
|-----------|-------------|
| **Architecture** | Clean separation of concerns (Event, EventMemory, ingestion functions) |
| **Real-time Audio** | Proper streaming with VAD, amplitude gating, and chunk-based processing |
| **Event Model** | Flexible multimodal event structure with timestamp and semantics |
| **Memory Bounds** | Prevents unbounded memory growth with max_events |
| **Intent Detection** | Reasonable keyword-based intent classification for MVP |
| **LLM Integration** | Grounded prompting pattern with evidence injection |

### Critical Weaknesses ✗

---

## 1. OCR Failure on Stylized Content

**Problem**: Tesseract fails catastrophically on your Amul images.

```python
# What Tesseract sees on amul_4.jpeg ("India's most engaging couple!")
# Likely output: "" or garbled text like "Indras m0st engag ng coup e"
```

**Evidence from your images**:
- Amul ads use hand-drawn stylized fonts
- Mixed Hindi/English text
- Text on colored/textured backgrounds
- Creative typography (e.g., "Dish Wiश!", "बutter Vutteर")

**Impact**: Your retrieval is 100% dependent on OCR. No OCR = No retrieval.

---

## 2. Naive Lexical Retrieval

**Problem**: Token overlap matching is too brittle.

```python
# Current approach
def tokenize(text: str):
    return set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))

# Query: "What did the cricket ad say?"
# Tokens: {"what", "did", "the", "cricket", "say"}

# OCR from amul_10.jpg (Bahuballebaaz): "Bahuballebaaz Amul Reddy to eat"
# Tokens: {"bahuballebaaz", "amul", "reddy", "eat"}

# Overlap: 0 tokens → NO MATCH (even though it's clearly cricket-related)
```

**Failures**:
- Synonyms don't match ("cricket" vs "bat", "ball", "match")
- Paraphrases fail completely
- Requires exact word overlap

---

## 3. No Visual Understanding

**Problem**: If OCR fails, you have zero information about the image.

```python
# For amul_7.jpeg (political cartoon with multiple people around a table)
# OCR might get: "The son is set to rise! Amul Utterly Beta-ly Delicious!"
# 
# But you lose:
# - There are 5+ people in the image
# - They're sitting around food
# - One person appears prominent (the "son")
# - Political/satirical context
```

---

## 4. No Persistence

**Problem**: Memory dies with the process.

```python
# After Ctrl+C:
memory = EventMemory(max_events=50)  # Everything gone
```

---

## 5. No Speaker Identity

**Problem**: You transcribe but don't know WHO spoke.

```python
# Current semantic for voice:
{
    "transcript": "What does the cricket image say?",
    "intent": "ask_image",
    # No speaker_id, no voice embedding
}

# If Alice and Bob both talk, you can't tell them apart
```

---

## 6. Single-Modality Events

**Problem**: Events are created per-modality, not unified.

```python
# If user shows image while speaking about it:
# Event 1: voice event (transcript)
# Event 2: image event (OCR)
# 
# No link between them!
```

---

## Part 2: Step-by-Step Improvement Plan

### Priority Order (Impact vs Effort)

```
HIGH IMPACT, LOW EFFORT (Do First)
├── 1. Add image captioning fallback
├── 2. Add embedding-based retrieval
└── 3. Add persistence (SQLite + pickle)

HIGH IMPACT, MEDIUM EFFORT (Do Second)  
├── 4. Improve OCR with EasyOCR
├── 5. Add cross-modal event linking
└── 6. Add confidence-based response routing

MEDIUM IMPACT, MEDIUM EFFORT (Do Third)
├── 7. Add speaker embedding
├── 8. Add semantic chunking
└── 9. Add memory consolidation

FUTURE (Skip for now)
├── Graph database
├── ColPali visual retrieval
└── Full diarization pipeline
```

---

## Step 1: Add Image Captioning Fallback

**Why**: When OCR fails (empty or low confidence), get visual description instead.

**Implementation**:

```python
# image_ingest_v2.py

from PIL import Image
import pytesseract
import datetime
import os
import requests
import base64

def get_image_caption(image_path: str) -> str:
    """Use local VLM (LLaVA via Ollama) to caption image"""
    
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": "llava",  # or "llava:13b" for better quality
            "prompt": "Describe this image in detail. Include any text you can see, people, objects, colors, and the overall scene.",
            "images": [image_b64],
            "stream": False
        }
    )
    
    data = response.json()
    return data.get("response", "").strip()


def ingest_image(image_path: str) -> dict:
    """
    Dual-path image ingestion:
    1. Try OCR first
    2. If OCR weak, add visual caption
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    image = Image.open(image_path)
    
    # Path 1: OCR
    ocr_text = pytesseract.image_to_string(image).strip()
    ocr_confidence = calculate_ocr_confidence(ocr_text)
    
    # Path 2: Visual caption (if OCR is weak)
    caption = ""
    if ocr_confidence < 0.5 or len(ocr_text) < 10:
        caption = get_image_caption(image_path)
    
    semantic = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "path": image_path,
        "ocr_text": ocr_text,
        "caption": caption,
        "combined_text": f"{ocr_text}\n{caption}".strip(),
        "ocr_confidence": ocr_confidence,
        "has_caption": bool(caption)
    }

    return semantic


def calculate_ocr_confidence(ocr_text: str) -> float:
    """
    Heuristic OCR confidence based on text quality.
    """
    if not ocr_text:
        return 0.0
    
    # Factors that indicate good OCR:
    word_count = len(ocr_text.split())
    avg_word_len = sum(len(w) for w in ocr_text.split()) / max(word_count, 1)
    alpha_ratio = sum(c.isalpha() for c in ocr_text) / max(len(ocr_text), 1)
    
    # Score components
    length_score = min(word_count / 10, 1.0)  # More words = better
    word_len_score = 1.0 if 3 < avg_word_len < 10 else 0.5  # Reasonable word length
    alpha_score = alpha_ratio  # More letters vs garbage
    
    return (length_score + word_len_score + alpha_score) / 3
```

**Test with your Amul images**:
```python
# amul_7.jpeg - "The son is set to rise!"
# OCR might get partial text
# Caption will get: "A cartoon showing several people sitting around a table 
#                   with food. The Amul butter girl is present. Text reads 
#                   'The son is set to rise' suggesting a political commentary..."
```

---

## Step 2: Add Embedding-Based Retrieval

**Why**: Semantic similarity instead of exact word match.

**Implementation**:

```python
# retrieval_v2.py

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple

# Load once at startup
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality

class EmbeddingIndex:
    """Simple in-memory embedding index"""
    
    def __init__(self):
        self.embeddings = []  # List of numpy arrays
        self.event_ids = []   # Corresponding event IDs
        self.texts = []       # Original texts (for debugging)
    
    def add(self, event_id: str, text: str):
        if not text.strip():
            return
        
        embedding = embedder.encode(text, normalize_embeddings=True)
        self.embeddings.append(embedding)
        self.event_ids.append(event_id)
        self.texts.append(text)
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Returns list of (event_id, similarity_score)"""
        if not self.embeddings:
            return []
        
        query_emb = embedder.encode(query, normalize_embeddings=True)
        
        # Compute cosine similarities
        embeddings_matrix = np.array(self.embeddings)
        similarities = embeddings_matrix @ query_emb
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Minimum threshold
                results.append((self.event_ids[idx], float(similarities[idx])))
        
        return results


# Integration with EventMemory
class EventMemoryV2:
    def __init__(self, max_events: int = 100):
        self.events: dict = {}  # event_id -> Event
        self.event_order: list = []  # For recency
        self.max_events = max_events
        self.embedding_index = EmbeddingIndex()

    def add_event(self, event):
        self.events[event.event_id] = event
        self.event_order.append(event.event_id)
        
        # Index the text content
        text_content = self._extract_text(event)
        if text_content:
            self.embedding_index.add(event.event_id, text_content)
        
        # Bound memory
        if len(self.event_order) > self.max_events:
            old_id = self.event_order.pop(0)
            del self.events[old_id]
            # Note: embeddings not removed (acceptable for MVP)

    def _extract_text(self, event) -> str:
        """Extract searchable text from any event type"""
        texts = []
        
        if "image" in event.semantics:
            img = event.semantics["image"]
            texts.append(img.get("combined_text", ""))
            texts.append(img.get("ocr_text", ""))
            texts.append(img.get("caption", ""))
        
        if "voice" in event.semantics:
            texts.append(event.semantics["voice"].get("transcript", ""))
        
        return " ".join(filter(None, texts))

    def semantic_search(self, query: str, k: int = 5) -> list:
        """Find events by semantic similarity"""
        results = self.embedding_index.search(query, k)
        return [self.events[eid] for eid, score in results if eid in self.events]
```

**Why this matters**:
```python
# Query: "Show me the cricket advertisement"
# 
# Lexical (old): Needs exact word "cricket" in OCR
# Semantic (new): Finds images with "bat", "ball", "match", "player", etc.
#
# amul_10.jpg caption: "A cartoon showing a cricket player with bat and pads, 
#                       the Amul girl offering butter..."
# Similarity to "cricket advertisement": 0.72 ✓
```

---

## Step 3: Add Persistence (SQLite)

**Why**: Don't lose memory on restart.

**Implementation**:

```python
# persistence.py

import sqlite3
import json
import pickle
from pathlib import Path
from typing import Optional

class MemoryStore:
    """SQLite-based persistent storage for events"""
    
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                modalities TEXT NOT NULL,
                semantics_json TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def save_event(self, event, embedding: Optional[list] = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_blob = pickle.dumps(embedding) if embedding else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO events 
            (event_id, timestamp, modalities, semantics_json, embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (
            event.event_id,
            event.timestamp,
            json.dumps(event.modalities),
            json.dumps(event.semantics),
            embedding_blob
        ))
        
        conn.commit()
        conn.close()
    
    def load_all_events(self) -> list:
        """Load all events from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM events ORDER BY timestamp")
        rows = cursor.fetchall()
        conn.close()
        
        events = []
        for row in rows:
            event_id, timestamp, modalities_json, semantics_json, embedding_blob, created_at = row
            
            # Reconstruct Event object
            event = Event.__new__(Event)
            event.event_id = event_id
            event.timestamp = timestamp
            event.modalities = json.loads(modalities_json)
            event.semantics = json.loads(semantics_json)
            
            events.append(event)
        
        return events
    
    def load_recent(self, n: int = 50) -> list:
        """Load n most recent events"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM events 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (n,))
        
        rows = cursor.fetchall()
        conn.close()
        
        # ... reconstruct events ...
        return events


# Updated EventMemory with persistence
class PersistentEventMemory(EventMemoryV2):
    def __init__(self, db_path: str = "memory.db", max_events: int = 100):
        super().__init__(max_events)
        self.store = MemoryStore(db_path)
        self._load_from_disk()
    
    def _load_from_disk(self):
        """Restore memory from database on startup"""
        events = self.store.load_recent(self.max_events)
        for event in events:
            # Add to in-memory structures without re-saving
            self.events[event.event_id] = event
            self.event_order.append(event.event_id)
            
            text_content = self._extract_text(event)
            if text_content:
                self.embedding_index.add(event.event_id, text_content)
        
        print(f"Loaded {len(events)} events from disk")
    
    def add_event(self, event):
        super().add_event(event)
        
        # Persist to disk
        text_content = self._extract_text(event)
        embedding = embedder.encode(text_content).tolist() if text_content else None
        self.store.save_event(event, embedding)
```

---

## Step 4: Improve OCR with EasyOCR

**Why**: Better multilingual support, handles stylized text better.

```python
# ocr_improved.py

import easyocr
from PIL import Image
import numpy as np

# Initialize once (supports Hindi + English)
reader = easyocr.Reader(['en', 'hi'], gpu=False)  # Set gpu=True if available

def extract_text_easyocr(image_path: str) -> dict:
    """
    EasyOCR extraction with bounding boxes and confidence.
    """
    results = reader.readtext(image_path)
    
    # results = [(bbox, text, confidence), ...]
    
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
        "raw_results": results  # Keep for debugging
    }


def hybrid_ocr(image_path: str) -> dict:
    """
    Try multiple OCR engines and combine results.
    """
    import pytesseract
    from PIL import Image
    
    # Engine 1: Tesseract
    image = Image.open(image_path)
    tess_text = pytesseract.image_to_string(image).strip()
    
    # Engine 2: EasyOCR
    easy_result = extract_text_easyocr(image_path)
    
    # Combine: prefer EasyOCR if it found more
    if len(easy_result["text"]) > len(tess_text) * 1.2:
        primary = easy_result["text"]
        confidence = easy_result["confidence"]
    else:
        primary = tess_text
        confidence = 0.7 if tess_text else 0.0
    
    return {
        "text": primary,
        "confidence": confidence,
        "tesseract_text": tess_text,
        "easyocr_text": easy_result["text"]
    }
```

**Expected improvement on your images**:

| Image | Tesseract | EasyOCR |
|-------|-----------|---------|
| amul_4.jpeg | "Indias most..." (partial) | "India's most engaging couple! Amul Captain of butlers" |
| amul_9.jpg | "" (fails on Hindi) | "Dish Wiश! Amul बutter Vutteर" |
| amul_7.jpeg | "The son is..." | "The son is set to rise! Amul Utterly Beta-ly Delicious!" |

---

## Step 5: Cross-Modal Event Linking

**Why**: Connect voice and image events that occur together.

```python
# event_linking.py

from datetime import datetime, timedelta

class LinkedEventMemory(PersistentEventMemory):
    """Memory that links related events across modalities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_window = timedelta(seconds=10)  # Link events within 10s
    
    def add_event(self, event):
        # Find recent events in other modalities to link
        linked_events = self._find_temporal_neighbors(event)
        
        # Store links bidirectionally
        if linked_events:
            event.linked_event_ids = [e.event_id for e in linked_events]
            for linked in linked_events:
                if not hasattr(linked, 'linked_event_ids'):
                    linked.linked_event_ids = []
                linked.linked_event_ids.append(event.event_id)
        
        super().add_event(event)
    
    def _find_temporal_neighbors(self, new_event) -> list:
        """Find events within temporal window in different modalities"""
        new_time = datetime.fromisoformat(new_event.timestamp)
        neighbors = []
        
        for event_id in reversed(self.event_order[-20:]):  # Check recent
            event = self.events.get(event_id)
            if not event:
                continue
            
            # Skip same modality
            if set(event.modalities) & set(new_event.modalities):
                continue
            
            event_time = datetime.fromisoformat(event.timestamp)
            if abs(new_time - event_time) <= self.temporal_window:
                neighbors.append(event)
        
        return neighbors
    
    def get_context_for_event(self, event) -> dict:
        """Get full context including linked events"""
        context = {
            "primary": event,
            "linked": []
        }
        
        if hasattr(event, 'linked_event_ids'):
            for eid in event.linked_event_ids:
                if eid in self.events:
                    context["linked"].append(self.events[eid])
        
        return context
```

**Usage**:
```python
# User shows image while saying "What does this say?"
# 
# voice_event links to image_event
# 
# When processing voice query, we automatically get the image context
```

---

## Step 6: Confidence-Based Response Routing

**Why**: Choose best response strategy based on data quality.

```python
# response_router.py

def generate_response(event, memory):
    """
    Route to best response strategy based on confidence.
    """
    query = event.semantics["voice"]["transcript"]
    intent = event.semantics["voice"]["intent"]
    
    # Get linked context (cross-modal)
    context = memory.get_context_for_event(event)
    
    # Get retrieved context (semantic search)
    retrieved = memory.semantic_search(query, k=3)
    
    # Determine best response strategy
    strategy = determine_strategy(context, retrieved, intent)
    
    if strategy == "DIRECT_READ":
        # High confidence OCR - just read it
        return direct_read_response(context, retrieved)
    
    elif strategy == "CAPTION_DESCRIBE":
        # OCR failed but caption available
        return caption_based_response(context, retrieved)
    
    elif strategy == "LLM_GROUND":
        # Need LLM to synthesize
        return llm_grounded_response(query, context, retrieved)
    
    elif strategy == "CLARIFY":
        # Not enough info
        return "I'm not sure what you're referring to. Could you point to the specific image or give me more context?"
    
    else:
        return "I don't have enough information to answer that."


def determine_strategy(context, retrieved, intent) -> str:
    """
    Decision logic for response strategy.
    """
    has_linked_image = any(
        "image" in e.modalities 
        for e in context.get("linked", [])
    )
    
    has_retrieved = len(retrieved) > 0
    
    # Check OCR/caption quality
    best_ocr_conf = 0
    has_caption = False
    
    all_images = context.get("linked", []) + retrieved
    for e in all_images:
        if "image" in e.modalities:
            img_sem = e.semantics.get("image", {})
            best_ocr_conf = max(best_ocr_conf, img_sem.get("ocr_confidence", 0))
            if img_sem.get("caption"):
                has_caption = True
    
    # Decision tree
    if intent == "ask_image":
        if has_linked_image and best_ocr_conf > 0.7:
            return "DIRECT_READ"
        elif has_linked_image and has_caption:
            return "CAPTION_DESCRIBE"
        elif has_retrieved:
            return "LLM_GROUND"
        else:
            return "CLARIFY"
    
    elif intent == "reference":
        if has_linked_image:
            return "LLM_GROUND"
        else:
            return "CLARIFY"
    
    else:
        if has_retrieved:
            return "LLM_GROUND"
        return "UNKNOWN"
```

---

## Part 3: Improved Architecture

After implementing Steps 1-6:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IMPROVED ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐     ┌──────────────┐                            │
│   │  Audio Input │     │ Image Input  │                            │
│   └──────┬───────┘     └──────┬───────┘                            │
│          │                    │                                     │
│          ▼                    ▼                                     │
│   ┌──────────────┐     ┌──────────────┐                            │
│   │   Whisper    │     │ Hybrid OCR   │                            │
│   │   (ASR)      │     │ (Tess+Easy)  │                            │
│   └──────┬───────┘     └──────┬───────┘                            │
│          │                    │                                     │
│          │                    ▼                                     │
│          │             ┌──────────────┐                            │
│          │             │ If OCR weak: │                            │
│          │             │ VLM Caption  │                            │
│          │             └──────┬───────┘                            │
│          │                    │                                     │
│          ▼                    ▼                                     │
│   ┌──────────────────────────────────────┐                         │
│   │         Event Creation               │                         │
│   │   + Temporal Cross-Modal Linking     │                         │
│   └──────────────────┬───────────────────┘                         │
│                      │                                              │
│          ┌───────────┼───────────┐                                 │
│          ▼           ▼           ▼                                 │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐                          │
│   │ In-Memory│ │ Embedding│ │ SQLite   │                          │
│   │ Events   │ │  Index   │ │ Persist  │                          │
│   └──────────┘ └──────────┘ └──────────┘                          │
│          │           │           │                                 │
│          └───────────┼───────────┘                                 │
│                      ▼                                              │
│            ┌──────────────────┐                                    │
│            │ Semantic Search  │                                    │
│            │ (Embedding-based)│                                    │
│            └────────┬─────────┘                                    │
│                     ▼                                              │
│            ┌──────────────────┐                                    │
│            │ Response Router  │                                    │
│            │ (Strategy Select)│                                    │
│            └────────┬─────────┘                                    │
│                     │                                              │
│      ┌──────────────┼──────────────┐                              │
│      ▼              ▼              ▼                              │
│ ┌─────────┐   ┌─────────┐   ┌─────────┐                          │
│ │ Direct  │   │ Caption │   │  LLM    │                          │
│ │  Read   │   │ Describe│   │ Ground  │                          │
│ └─────────┘   └─────────┘   └─────────┘                          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Implementation Order & Timeline

### Week 1: Foundation Fixes

```
Day 1-2: Image Captioning Fallback
├── Install: ollama pull llava
├── Implement: get_image_caption()
├── Integrate: Update ingest_image()
└── Test: Run on all Amul images

Day 3-4: Embedding Retrieval
├── Install: pip install sentence-transformers
├── Implement: EmbeddingIndex class
├── Integrate: Update EventMemory
└── Test: Query "cricket ad" finds amul_10.jpg

Day 5-7: Persistence
├── Implement: MemoryStore with SQLite
├── Implement: PersistentEventMemory
├── Test: Restart process, verify memory loads
└── Test: Add 100 events, verify bounds work
```

### Week 2: Quality Improvements

```
Day 1-2: Better OCR
├── Install: pip install easyocr
├── Implement: hybrid_ocr()
├── Test: Compare Tesseract vs EasyOCR on Amul images
└── Integrate: Update ingest_image()

Day 3-4: Cross-Modal Linking
├── Implement: LinkedEventMemory
├── Test: Show image + speak, verify link
└── Update: Response generation to use links

Day 5-7: Response Router
├── Implement: determine_strategy()
├── Implement: Strategy-specific responses
├── Test: Verify correct routing
└── End-to-end testing
```

---

## Part 5: Test Cases

### Test 1: OCR Fallback
```python
# Input: amul_4.jpeg (stylized text)
# Expected: 
#   - OCR confidence < 0.5
#   - Caption generated: "A cartoon showing two people (likely celebrities) 
#     on a beach with cricket equipment. Text reads 'India's most engaging couple'..."
#   - combined_text contains both OCR and caption
```

### Test 2: Semantic Retrieval
```python
# Memory contains: amul_10.jpg ("Bahuballebaaz - cricket player")
# Query: "Show me the sports advertisement"
# Expected: amul_10.jpg retrieved with similarity > 0.5
```

### Test 3: Cross-Modal Linking
```python
# Action: User shows image, says "What does this say?" within 5 seconds
# Expected: voice_event.linked_event_ids contains image_event.event_id
```

### Test 4: Persistence
```python
# Action: Add 10 events, kill process, restart
# Expected: All 10 events restored from SQLite
```

### Test 5: Response Strategy
```python
# Scenario A: High OCR confidence
# Expected: DIRECT_READ strategy, no LLM call

# Scenario B: Low OCR, good caption
# Expected: CAPTION_DESCRIBE strategy

# Scenario C: Complex query
# Expected: LLM_GROUND strategy with grounded prompt
```

---

## Part 6: Code Refactoring Suggestions

### Current Issues in voice_pipeline.py

1. **Global state**: `memory`, `stream`, `model` are global
2. **Duplicate classes**: Event/EventMemory defined twice
3. **Hardcoded paths**: `/Users/shubhgarg/Downloads/...`
4. **Mixed concerns**: Audio capture + processing + LLM in one file

### Suggested Structure

```
multimodal_memory/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── event.py           # Event, EventMemory classes
│   ├── persistence.py     # SQLite storage
│   └── retrieval.py       # EmbeddingIndex, semantic search
├── ingestion/
│   ├── __init__.py
│   ├── image.py           # Image ingestion (OCR + caption)
│   ├── audio.py           # Audio capture and transcription
│   └── ocr.py             # OCR backends
├── generation/
│   ├── __init__.py
│   ├── router.py          # Response strategy routing
│   ├── prompts.py         # Prompt templates
│   └── llm.py             # LLM interface (Ollama)
├── config.py              # All configuration
└── main.py                # Entry point
```

---

## Quick Wins (Do Today)

1. **Add debug print for OCR output**:
```python
def ingest_image(image_path: str):
    # ... existing code ...
    print(f"[DEBUG OCR] {image_path}")
    print(f"  Text: {ocr_text[:100]}...")
    print(f"  Confidence: {confidence}")
    return semantic
```

2. **Add fallback for empty retrieval**:
```python
def retrieve_relevant_images(query, memory, k=3):
    # ... existing code ...
    
    if not any(score > 0 for score, _ in scored):
        # IMPROVEMENT: Return most recent instead of nothing
        recent = [e for e in memory.events if "image" in e.modalities][-k:]
        if recent:
            return recent
        return []  # Only if truly no images
```

3. **Add image path to response**:
```python
def generate_response(event, memory):
    # ... existing code ...
    
    # Include which images were used
    image_paths = [e.semantics["image"]["path"] for e in image_events]
    return f"{response}\n\n[Sources: {', '.join(image_paths)}]"
```

---

## Summary

Your current system is a solid MVP. The main gaps are:

| Issue | Impact | Fix |
|-------|--------|-----|
| OCR fails on stylized text | Can't understand Amul ads | Add VLM captioning |
| Lexical retrieval | Misses semantic matches | Add embeddings |
| No persistence | Lose memory on restart | Add SQLite |
| No cross-modal links | Can't connect speech to shown image | Add temporal linking |

Follow the 2-week plan above, and you'll have a significantly more capable system that can actually understand and retrieve your Amul advertisement images.
