"""
Integrated Event Memory System
Combines:
- Event storage (in-memory + persistent)
- Embedding-based retrieval
- Cross-modal linking
- Response routing
"""

import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# Import our modules
from persistence import MemoryStore, reconstruct_event
from retrieval_v2 import EmbeddingIndex, HybridRetriever, SearchResult


# ============================================================
# Core Event Class
# ============================================================

class Event:
    """
    Multimodal event container.
    Can hold data from multiple modalities (image, voice, text).
    """
    
    def __init__(self, modality: str, semantic: Dict):
        self.event_id = str(uuid.uuid4())
        self.timestamp = semantic.get("timestamp", datetime.utcnow().isoformat())
        self.modalities = [modality]
        self.semantics = {modality: semantic}
        self.linked_event_ids: List[str] = []
    
    def add_modality(self, modality: str, semantic: Dict):
        """Add another modality to this event"""
        if modality not in self.modalities:
            self.modalities.append(modality)
        self.semantics[modality] = semantic
    
    def get_text_content(self) -> str:
        """Extract all text content for indexing"""
        texts = []
        
        if "image" in self.semantics:
            img = self.semantics["image"]
            texts.append(img.get("combined_text", ""))
            texts.append(img.get("ocr_text", ""))
            texts.append(img.get("caption", ""))
        
        if "voice" in self.semantics:
            texts.append(self.semantics["voice"].get("transcript", ""))
        
        if "text" in self.semantics:
            texts.append(self.semantics["text"].get("content", ""))
        
        return " ".join(filter(None, texts))
    
    def __repr__(self):
        return f"<Event {self.event_id[:8]} | {self.modalities} | {self.timestamp}>"


# ============================================================
# Integrated Memory System
# ============================================================

class MultimodalMemory:
    """
    Complete multimodal memory system with:
    - Persistent storage (SQLite)
    - Semantic retrieval (embeddings)
    - Cross-modal linking
    - Bounded memory management
    """
    
    def __init__(
        self,
        db_path: str = "memory.db",
        max_events: int = 100,
        embedding_model: str = "all-MiniLM-L6-v2",
        temporal_window_seconds: int = 10
    ):
        """
        Initialize the memory system.
        
        Args:
            db_path: Path to SQLite database
            max_events: Maximum events to keep in memory
            embedding_model: Sentence transformer model name
            temporal_window_seconds: Window for cross-modal linking
        """
        self.max_events = max_events
        self.temporal_window = timedelta(seconds=temporal_window_seconds)
        
        # Storage components
        self.store = MemoryStore(db_path)
        self.embedding_index = EmbeddingIndex(embedding_model)
        
        # In-memory cache
        self.events: Dict[str, Event] = {}
        self.event_order: List[str] = []  # For recency tracking
        
        # Load existing data
        self._load_from_disk()
    
    def _load_from_disk(self):
        """Restore memory from database on startup"""
        stored = self.store.load_recent(self.max_events)
        
        for event_dict in stored:
            # Reconstruct event
            event = reconstruct_event(event_dict, Event)
            
            # Load links
            event.linked_event_ids = self.store.load_linked_events(event.event_id)
            
            # Add to memory (without re-saving)
            self.events[event.event_id] = event
            self.event_order.append(event.event_id)
            
            # Index for retrieval
            text = event.get_text_content()
            if text:
                self.embedding_index.add(event.event_id, text)
        
        if stored:
            print(f"[MEMORY] Loaded {len(stored)} events from disk")
    
    def add_event(self, event: Event) -> Event:
        """
        Add an event to memory.
        - Links to temporally close events
        - Indexes for retrieval
        - Persists to disk
        - Manages memory bounds
        """
        # Step 1: Find and create cross-modal links
        linked = self._find_temporal_neighbors(event)
        for linked_event in linked:
            event.linked_event_ids.append(linked_event.event_id)
            linked_event.linked_event_ids.append(event.event_id)
            
            # Persist link
            self.store.save_link(event.event_id, linked_event.event_id, "temporal")
        
        # Step 2: Add to in-memory structures
        self.events[event.event_id] = event
        self.event_order.append(event.event_id)
        
        # Step 3: Index for retrieval
        text = event.get_text_content()
        if text:
            self.embedding_index.add(event.event_id, text)
        
        # Step 4: Persist to disk
        embedding = None
        if text:
            # Get embedding for storage
            self.embedding_index._ensure_model()
            embedding = self.embedding_index.model.encode(text, normalize_embeddings=True).tolist()
        
        self.store.save_event(event, embedding)
        
        # Step 5: Enforce memory bounds
        self._enforce_bounds()
        
        return event
    
    def _find_temporal_neighbors(self, new_event: Event) -> List[Event]:
        """Find events within temporal window in different modalities"""
        try:
            new_time = datetime.fromisoformat(new_event.timestamp)
        except:
            new_time = datetime.utcnow()
        
        neighbors = []
        
        # Check recent events (last 20)
        for event_id in reversed(self.event_order[-20:]):
            event = self.events.get(event_id)
            if not event:
                continue
            
            # Skip if same modality
            if set(event.modalities) & set(new_event.modalities):
                continue
            
            try:
                event_time = datetime.fromisoformat(event.timestamp)
                if abs(new_time - event_time) <= self.temporal_window:
                    neighbors.append(event)
            except:
                continue
        
        return neighbors
    
    def _enforce_bounds(self):
        """Remove old events if over max"""
        while len(self.event_order) > self.max_events:
            old_id = self.event_order.pop(0)
            if old_id in self.events:
                del self.events[old_id]
            # Note: We don't remove from embedding index (acceptable overhead)
            # Database cleanup happens separately via store.delete_old_events()
    
    # ========================================
    # Retrieval Methods
    # ========================================
    
    def semantic_search(self, query: str, k: int = 5) -> List[Event]:
        """Find events by semantic similarity"""
        results = self.embedding_index.search(query, k=k, threshold=0.3)
        
        events = []
        for result in results:
            if result.event_id in self.events:
                events.append(self.events[result.event_id])
        
        return events
    
    def semantic_search_with_scores(self, query: str, k: int = 5) -> List[Tuple[Event, float]]:
        """Find events with similarity scores"""
        results = self.embedding_index.search(query, k=k, threshold=0.3)
        
        events = []
        for result in results:
            if result.event_id in self.events:
                events.append((self.events[result.event_id], result.score))
        
        return events
    
    def get_recent_events(self, n: int = 5) -> List[Event]:
        """Get n most recent events"""
        recent_ids = self.event_order[-n:]
        return [self.events[eid] for eid in recent_ids if eid in self.events]
    
    def get_events_by_modality(self, modality: str, n: int = 10) -> List[Event]:
        """Get events containing a specific modality"""
        events = []
        for event_id in reversed(self.event_order):
            event = self.events.get(event_id)
            if event and modality in event.modalities:
                events.append(event)
                if len(events) >= n:
                    break
        return events
    
    def get_linked_events(self, event: Event) -> List[Event]:
        """Get events linked to this one"""
        linked = []
        for eid in event.linked_event_ids:
            if eid in self.events:
                linked.append(self.events[eid])
        return linked
    
    def get_context_for_query(self, query: str, voice_event: Optional[Event] = None) -> Dict:
        """
        Get comprehensive context for answering a query.
        Combines linked events + semantic search.
        """
        context = {
            "linked": [],
            "retrieved": [],
            "all_events": []
        }
        
        # If we have a voice event, get linked context (e.g., shown images)
        if voice_event:
            context["linked"] = self.get_linked_events(voice_event)
        
        # Semantic search for relevant past events
        context["retrieved"] = self.semantic_search(query, k=5)
        
        # Combine and deduplicate
        seen_ids = set()
        for event in context["linked"] + context["retrieved"]:
            if event.event_id not in seen_ids:
                context["all_events"].append(event)
                seen_ids.add(event.event_id)
        
        return context
    
    # ========================================
    # Utility Methods
    # ========================================
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """Get a specific event by ID"""
        return self.events.get(event_id)
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        db_stats = self.store.get_stats()
        
        return {
            "in_memory_events": len(self.events),
            "indexed_embeddings": len(self.embedding_index),
            **db_stats
        }
    
    def cleanup(self, keep_count: int = None):
        """Clean up old events from database"""
        keep = keep_count or self.max_events
        deleted = self.store.delete_old_events(keep)
        print(f"[MEMORY] Deleted {deleted} old events from database")
        return deleted


# ============================================================
# Response Router
# ============================================================

@dataclass
class ResponseContext:
    """Container for response generation context"""
    query: str
    intent: str
    linked_images: List[Event] = field(default_factory=list)
    retrieved_events: List[Event] = field(default_factory=list)
    best_ocr_confidence: float = 0.0
    has_caption: bool = False
    strategy: str = "UNKNOWN"


class ResponseRouter:
    """
    Determines the best response strategy based on available context.
    """
    
    @staticmethod
    def analyze_context(
        query: str,
        intent: str,
        context: Dict
    ) -> ResponseContext:
        """Analyze available context and determine strategy"""
        
        rc = ResponseContext(query=query, intent=intent)
        
        # Separate by modality
        for event in context.get("linked", []):
            if "image" in event.modalities:
                rc.linked_images.append(event)
        
        rc.retrieved_events = context.get("retrieved", [])
        
        # Analyze image quality
        all_images = rc.linked_images + [
            e for e in rc.retrieved_events if "image" in e.modalities
        ]
        
        for event in all_images:
            img_sem = event.semantics.get("image", {})
            rc.best_ocr_confidence = max(
                rc.best_ocr_confidence,
                img_sem.get("ocr_confidence", 0)
            )
            if img_sem.get("caption"):
                rc.has_caption = True
        
        # Determine strategy
        rc.strategy = ResponseRouter._determine_strategy(rc)
        
        return rc
    
    @staticmethod
    def _determine_strategy(rc: ResponseContext) -> str:
        """Decision logic for response strategy"""
        
        if rc.intent in ["ask_image", "reference"]:
            # User is asking about an image
            if rc.linked_images and rc.best_ocr_confidence > 0.6:
                return "DIRECT_READ"
            elif rc.linked_images and rc.has_caption:
                return "CAPTION_DESCRIBE"
            elif rc.retrieved_events:
                return "LLM_GROUND"
            else:
                return "CLARIFY"
        
        elif rc.intent == "command":
            return "EXECUTE"
        
        else:
            # General query
            if rc.retrieved_events:
                return "LLM_GROUND"
            return "NO_CONTEXT"


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    import tempfile
    import os
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    print(f"Testing MultimodalMemory with: {db_path}")
    
    # Initialize memory
    memory = MultimodalMemory(db_path=db_path, max_events=50)
    
    # Add some test events
    print("\n--- Adding Events ---")
    
    # Image events
    img1 = Event("image", {
        "timestamp": datetime.utcnow().isoformat(),
        "path": "/test/amul_cricket.jpg",
        "ocr_text": "Bahuballebaaz Amul Reddy to eat",
        "caption": "A cartoon showing a cricket player with bat and pads, the Amul girl offering butter",
        "combined_text": "Bahuballebaaz Amul Reddy to eat. A cartoon showing a cricket player",
        "ocr_confidence": 0.7
    })
    memory.add_event(img1)
    print(f"Added: {img1}")
    
    img2 = Event("image", {
        "timestamp": datetime.utcnow().isoformat(),
        "path": "/test/amul_politics.jpg",
        "ocr_text": "The son is set to rise",
        "caption": "Political cartoon with people around a dining table",
        "combined_text": "The son is set to rise. Political cartoon with people",
        "ocr_confidence": 0.8
    })
    memory.add_event(img2)
    print(f"Added: {img2}")
    
    # Voice event (should link to recent image)
    import time
    time.sleep(0.1)  # Small delay
    
    voice1 = Event("voice", {
        "timestamp": datetime.utcnow().isoformat(),
        "transcript": "What does this cricket image say?",
        "intent": "ask_image"
    })
    memory.add_event(voice1)
    print(f"Added: {voice1}")
    print(f"  Linked to: {[eid[:8] for eid in voice1.linked_event_ids]}")
    
    # Test retrieval
    print("\n--- Testing Retrieval ---")
    
    query = "sports advertisement"
    results = memory.semantic_search_with_scores(query, k=3)
    print(f"Query: '{query}'")
    for event, score in results:
        print(f"  {event.event_id[:8]}: {score:.3f} - {event.modalities}")
    
    query = "political cartoon"
    results = memory.semantic_search_with_scores(query, k=3)
    print(f"Query: '{query}'")
    for event, score in results:
        print(f"  {event.event_id[:8]}: {score:.3f} - {event.modalities}")
    
    # Test context retrieval
    print("\n--- Testing Context Retrieval ---")
    context = memory.get_context_for_query("tell me about the cricket ad", voice1)
    print(f"Linked events: {len(context['linked'])}")
    print(f"Retrieved events: {len(context['retrieved'])}")
    
    # Test response routing
    print("\n--- Testing Response Router ---")
    rc = ResponseRouter.analyze_context(
        "what does this say",
        "ask_image",
        context
    )
    print(f"Strategy: {rc.strategy}")
    print(f"Best OCR confidence: {rc.best_ocr_confidence:.2f}")
    print(f"Has caption: {rc.has_caption}")
    
    # Test stats
    print("\n--- Memory Stats ---")
    stats = memory.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Test persistence (restart simulation)
    print("\n--- Testing Persistence ---")
    del memory
    
    memory2 = MultimodalMemory(db_path=db_path)
    print(f"After reload: {len(memory2.events)} events in memory")
    
    # Verify events restored
    for event in memory2.get_recent_events(5):
        print(f"  {event.event_id[:8]}: {event.modalities}")
    
    # Cleanup
    os.unlink(db_path)
    print("\nTest completed!")
