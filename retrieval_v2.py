"""
Embedding-Based Retrieval Module
- Semantic similarity search using sentence transformers
- In-memory vector index
- Hybrid retrieval (embedding + recency)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import json

# Try to import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("[WARN] sentence-transformers not installed. Run: pip install sentence-transformers")


@dataclass
class SearchResult:
    """Container for search results"""
    event_id: str
    score: float
    text: str
    event: Any = None


class EmbeddingIndex:
    """
    In-memory embedding index for semantic search.
    Uses sentence-transformers for encoding.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding index.
        
        Args:
            model_name: SentenceTransformer model to use
                - "all-MiniLM-L6-v2": Fast, good quality (384 dims)
                - "all-mpnet-base-v2": Better quality, slower (768 dims)
                - "multi-qa-MiniLM-L6-cos-v1": Optimized for QA
        """
        self.model_name = model_name
        self.model = None  # Lazy load
        
        self.embeddings: List[np.ndarray] = []
        self.event_ids: List[str] = []
        self.texts: List[str] = []
        self.metadata: List[Dict] = []
        
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._matrix_dirty = True
    
    def _ensure_model(self):
        """Lazy load the model"""
        if self.model is None:
            if not EMBEDDINGS_AVAILABLE:
                raise RuntimeError("sentence-transformers not installed")
            print(f"[INFO] Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
    
    def add(
        self, 
        event_id: str, 
        text: str, 
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add text to the index.
        
        Args:
            event_id: Unique identifier for this entry
            text: Text to embed and index
            metadata: Optional metadata to store
        
        Returns:
            True if added successfully
        """
        if not text or not text.strip():
            return False
        
        self._ensure_model()
        
        # Generate embedding
        embedding = self.model.encode(
            text, 
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        self.embeddings.append(embedding)
        self.event_ids.append(event_id)
        self.texts.append(text)
        self.metadata.append(metadata or {})
        
        self._matrix_dirty = True
        return True
    
    def _build_matrix(self):
        """Build numpy matrix for efficient search"""
        if self._matrix_dirty and self.embeddings:
            self._embeddings_matrix = np.array(self.embeddings)
            self._matrix_dirty = False
    
    def search(
        self, 
        query: str, 
        k: int = 5, 
        threshold: float = 0.3
    ) -> List[SearchResult]:
        """
        Search for similar texts.
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity score
        
        Returns:
            List of SearchResult objects
        """
        if not self.embeddings:
            return []
        
        self._ensure_model()
        self._build_matrix()
        
        # Encode query
        query_embedding = self.model.encode(
            query, 
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Compute cosine similarities (dot product since normalized)
        similarities = self._embeddings_matrix @ query_embedding
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append(SearchResult(
                    event_id=self.event_ids[idx],
                    score=score,
                    text=self.texts[idx]
                ))
        
        return results
    
    def search_by_embedding(
        self, 
        embedding: np.ndarray, 
        k: int = 5, 
        threshold: float = 0.3
    ) -> List[SearchResult]:
        """Search using a pre-computed embedding"""
        if not self.embeddings:
            return []
        
        self._build_matrix()
        
        # Normalize if needed
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        similarities = self._embeddings_matrix @ embedding
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append(SearchResult(
                    event_id=self.event_ids[idx],
                    score=score,
                    text=self.texts[idx]
                ))
        
        return results
    
    def remove(self, event_id: str) -> bool:
        """Remove an entry by event_id"""
        try:
            idx = self.event_ids.index(event_id)
            del self.embeddings[idx]
            del self.event_ids[idx]
            del self.texts[idx]
            del self.metadata[idx]
            self._matrix_dirty = True
            return True
        except ValueError:
            return False
    
    def clear(self):
        """Clear all entries"""
        self.embeddings.clear()
        self.event_ids.clear()
        self.texts.clear()
        self.metadata.clear()
        self._embeddings_matrix = None
        self._matrix_dirty = True
    
    def __len__(self):
        return len(self.embeddings)
    
    def save(self, path: str):
        """Save index to file"""
        data = {
            "model_name": self.model_name,
            "embeddings": [e.tolist() for e in self.embeddings],
            "event_ids": self.event_ids,
            "texts": self.texts,
            "metadata": self.metadata
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str):
        """Load index from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.model_name = data.get("model_name", self.model_name)
        self.embeddings = [np.array(e) for e in data["embeddings"]]
        self.event_ids = data["event_ids"]
        self.texts = data["texts"]
        self.metadata = data.get("metadata", [{} for _ in self.texts])
        self._matrix_dirty = True


class HybridRetriever:
    """
    Combines embedding search with other signals:
    - Semantic similarity
    - Recency
    - Lexical matching (BM25-like)
    """
    
    def __init__(
        self, 
        embedding_index: EmbeddingIndex,
        semantic_weight: float = 0.6,
        recency_weight: float = 0.2,
        lexical_weight: float = 0.2
    ):
        self.index = embedding_index
        self.semantic_weight = semantic_weight
        self.recency_weight = recency_weight
        self.lexical_weight = lexical_weight
    
    def search(
        self, 
        query: str,
        events: List[Any],
        k: int = 5,
        event_text_fn=None
    ) -> List[Tuple[Any, float]]:
        """
        Hybrid search combining multiple signals.
        
        Args:
            query: Search query
            events: List of event objects
            k: Number of results
            event_text_fn: Function to extract text from event
        
        Returns:
            List of (event, score) tuples
        """
        if not events:
            return []
        
        # Build event lookup
        event_lookup = {}
        for event in events:
            event_lookup[event.event_id] = event
        
        # Score 1: Semantic similarity
        semantic_results = self.index.search(query, k=k*2, threshold=0.2)
        semantic_scores = {r.event_id: r.score for r in semantic_results}
        
        # Score 2: Recency (newer = higher)
        recency_scores = {}
        for i, event in enumerate(reversed(events)):
            # Linear decay from 1.0 to 0.0
            recency_scores[event.event_id] = max(0, 1.0 - i * 0.02)
        
        # Score 3: Lexical overlap
        query_tokens = set(query.lower().split())
        lexical_scores = {}
        for event in events:
            if event_text_fn:
                text = event_text_fn(event)
            else:
                text = self._extract_text(event)
            
            if text:
                text_tokens = set(text.lower().split())
                overlap = len(query_tokens & text_tokens)
                lexical_scores[event.event_id] = min(overlap / len(query_tokens), 1.0) if query_tokens else 0
        
        # Combine scores
        final_scores = {}
        for event_id in event_lookup:
            sem = semantic_scores.get(event_id, 0)
            rec = recency_scores.get(event_id, 0)
            lex = lexical_scores.get(event_id, 0)
            
            final_scores[event_id] = (
                self.semantic_weight * sem +
                self.recency_weight * rec +
                self.lexical_weight * lex
            )
        
        # Sort and return top-k
        sorted_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
        
        results = []
        for event_id in sorted_ids[:k]:
            if final_scores[event_id] > 0.1:  # Minimum threshold
                results.append((event_lookup[event_id], final_scores[event_id]))
        
        return results
    
    def _extract_text(self, event) -> str:
        """Default text extraction from event"""
        texts = []
        
        if hasattr(event, 'semantics'):
            if "image" in event.semantics:
                img = event.semantics["image"]
                texts.append(img.get("combined_text", ""))
                texts.append(img.get("ocr_text", ""))
                texts.append(img.get("caption", ""))
            
            if "voice" in event.semantics:
                texts.append(event.semantics["voice"].get("transcript", ""))
        
        return " ".join(filter(None, texts))


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    # Test the embedding index
    print("Testing EmbeddingIndex...")
    
    index = EmbeddingIndex()
    
    # Add some test documents
    docs = [
        ("doc1", "The cricket player hit the ball over the boundary"),
        ("doc2", "A delicious butter spread on fresh bread"),
        ("doc3", "The political leader addressed the nation"),
        ("doc4", "India won the cricket world cup final"),
        ("doc5", "Fresh dairy products from the cooperative"),
    ]
    
    for doc_id, text in docs:
        index.add(doc_id, text)
    
    print(f"\nIndexed {len(index)} documents")
    
    # Test searches
    queries = [
        "sports match",
        "dairy butter",
        "government speech",
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = index.search(query, k=3)
        for r in results:
            print(f"  {r.event_id}: {r.score:.3f} - {r.text[:50]}...")
