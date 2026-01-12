"""
Persistence Module
- SQLite-based storage for events
- Embedding storage with pickle
- Load/save functionality
"""

import sqlite3
import json
import pickle
from pathlib import Path
from typing import List, Optional, Any
from datetime import datetime
import os


class MemoryStore:
    """
    SQLite-based persistent storage for multimodal events.
    """
    
    def __init__(self, db_path: str = "memory.db"):
        """
        Initialize the memory store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create database schema if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                modalities TEXT NOT NULL,
                semantics_json TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Index for timestamp queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_timestamp 
            ON events(timestamp DESC)
        """)
        
        # Index for modality filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_modalities 
            ON events(modalities)
        """)
        
        # Links table for cross-modal connections
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS event_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_event_id TEXT NOT NULL,
                target_event_id TEXT NOT NULL,
                link_type TEXT DEFAULT 'temporal',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_event_id) REFERENCES events(event_id),
                FOREIGN KEY (target_event_id) REFERENCES events(event_id),
                UNIQUE(source_event_id, target_event_id, link_type)
            )
        """)
        
        # Metadata table for system state
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_event(
        self, 
        event, 
        embedding: Optional[list] = None
    ) -> bool:
        """
        Save an event to the database.
        
        Args:
            event: Event object with event_id, timestamp, modalities, semantics
            embedding: Optional embedding vector as list
        
        Returns:
            True if saved successfully
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            embedding_blob = pickle.dumps(embedding) if embedding else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO events 
                (event_id, timestamp, modalities, semantics_json, embedding, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.timestamp,
                json.dumps(event.modalities),
                json.dumps(event.semantics),
                embedding_blob,
                datetime.utcnow().isoformat()
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to save event: {e}")
            conn.rollback()
            return False
            
        finally:
            conn.close()
    
    def save_link(
        self, 
        source_id: str, 
        target_id: str, 
        link_type: str = "temporal"
    ) -> bool:
        """Save a link between two events"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO event_links 
                (source_event_id, target_event_id, link_type)
                VALUES (?, ?, ?)
            """, (source_id, target_id, link_type))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to save link: {e}")
            return False
            
        finally:
            conn.close()
    
    def load_event(self, event_id: str) -> Optional[dict]:
        """Load a single event by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT event_id, timestamp, modalities, semantics_json, embedding
            FROM events WHERE event_id = ?
        """, (event_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_dict(row)
        return None
    
    def load_recent(self, n: int = 50) -> List[dict]:
        """Load n most recent events"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT event_id, timestamp, modalities, semantics_json, embedding
            FROM events 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (n,))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Return in chronological order (oldest first)
        return [self._row_to_dict(row) for row in reversed(rows)]
    
    def load_by_modality(self, modality: str, limit: int = 100) -> List[dict]:
        """Load events containing a specific modality"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # SQLite JSON contains check
        cursor.execute("""
            SELECT event_id, timestamp, modalities, semantics_json, embedding
            FROM events 
            WHERE modalities LIKE ?
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (f'%"{modality}"%', limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in rows]
    
    def load_linked_events(self, event_id: str) -> List[str]:
        """Get IDs of events linked to this one"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT target_event_id FROM event_links WHERE source_event_id = ?
            UNION
            SELECT source_event_id FROM event_links WHERE target_event_id = ?
        """, (event_id, event_id))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [row[0] for row in rows]
    
    def load_all(self) -> List[dict]:
        """Load all events"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT event_id, timestamp, modalities, semantics_json, embedding
            FROM events 
            ORDER BY timestamp ASC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in rows]
    
    def delete_event(self, event_id: str) -> bool:
        """Delete an event and its links"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Delete links first
            cursor.execute("""
                DELETE FROM event_links 
                WHERE source_event_id = ? OR target_event_id = ?
            """, (event_id, event_id))
            
            # Delete event
            cursor.execute("DELETE FROM events WHERE event_id = ?", (event_id,))
            
            conn.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            print(f"[ERROR] Failed to delete event: {e}")
            conn.rollback()
            return False
            
        finally:
            conn.close()
    
    def delete_old_events(self, keep_count: int = 100) -> int:
        """Delete oldest events, keeping only the most recent"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get IDs to delete
            cursor.execute("""
                SELECT event_id FROM events 
                ORDER BY timestamp DESC 
                LIMIT -1 OFFSET ?
            """, (keep_count,))
            
            old_ids = [row[0] for row in cursor.fetchall()]
            
            if old_ids:
                placeholders = ','.join(['?' for _ in old_ids])
                
                # Delete links
                cursor.execute(f"""
                    DELETE FROM event_links 
                    WHERE source_event_id IN ({placeholders}) 
                       OR target_event_id IN ({placeholders})
                """, old_ids + old_ids)
                
                # Delete events
                cursor.execute(f"DELETE FROM events WHERE event_id IN ({placeholders})", old_ids)
            
            conn.commit()
            return len(old_ids)
            
        except Exception as e:
            print(f"[ERROR] Failed to delete old events: {e}")
            conn.rollback()
            return 0
            
        finally:
            conn.close()
    
    def get_stats(self) -> dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM events")
        event_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM event_links")
        link_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM events")
        time_range = cursor.fetchone()
        
        # Modality breakdown
        cursor.execute("SELECT modalities FROM events")
        modality_counts = {"image": 0, "voice": 0, "text": 0}
        for (mods,) in cursor.fetchall():
            for mod in json.loads(mods):
                modality_counts[mod] = modality_counts.get(mod, 0) + 1
        
        conn.close()
        
        return {
            "total_events": event_count,
            "total_links": link_count,
            "oldest_event": time_range[0],
            "newest_event": time_range[1],
            "modality_breakdown": modality_counts,
            "db_path": self.db_path,
            "db_size_mb": os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
        }
    
    def _row_to_dict(self, row) -> dict:
        """Convert database row to dictionary"""
        event_id, timestamp, modalities_json, semantics_json, embedding_blob = row
        
        return {
            "event_id": event_id,
            "timestamp": timestamp,
            "modalities": json.loads(modalities_json),
            "semantics": json.loads(semantics_json),
            "embedding": pickle.loads(embedding_blob) if embedding_blob else None
        }
    
    def set_metadata(self, key: str, value: Any):
        """Store arbitrary metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO metadata (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (key, json.dumps(value), datetime.utcnow().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        return default


# ============================================================
# Event reconstruction helper
# ============================================================

def reconstruct_event(event_dict: dict, event_class):
    """
    Reconstruct an Event object from stored dictionary.
    
    Args:
        event_dict: Dictionary from database
        event_class: The Event class to instantiate
    
    Returns:
        Reconstructed Event object
    """
    event = event_class.__new__(event_class)
    event.event_id = event_dict["event_id"]
    event.timestamp = event_dict["timestamp"]
    event.modalities = event_dict["modalities"]
    event.semantics = event_dict["semantics"]
    
    # Optional attributes
    if "linked_event_ids" in event_dict:
        event.linked_event_ids = event_dict["linked_event_ids"]
    
    return event


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    import tempfile
    import uuid
    
    # Create a simple Event class for testing
    class TestEvent:
        def __init__(self, modality: str, text: str):
            self.event_id = str(uuid.uuid4())
            self.timestamp = datetime.utcnow().isoformat()
            self.modalities = [modality]
            self.semantics = {modality: {"text": text}}
    
    # Test with temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    print(f"Testing with database: {db_path}")
    store = MemoryStore(db_path)
    
    # Add some events
    events = [
        TestEvent("image", "A picture of a sunset"),
        TestEvent("voice", "Hello, how are you?"),
        TestEvent("image", "Cricket match photo"),
        TestEvent("voice", "Tell me about the image"),
    ]
    
    for event in events:
        store.save_event(event, embedding=[0.1, 0.2, 0.3])
        print(f"Saved: {event.event_id[:8]}")
    
    # Add a link
    store.save_link(events[2].event_id, events[3].event_id, "temporal")
    
    # Load and verify
    print("\nLoaded recent events:")
    loaded = store.load_recent(10)
    for e in loaded:
        print(f"  {e['event_id'][:8]}: {e['modalities']}")
    
    # Get stats
    print("\nStats:")
    stats = store.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Load by modality
    print("\nImage events:")
    images = store.load_by_modality("image")
    for e in images:
        print(f"  {e['event_id'][:8]}")
    
    # Get links
    print(f"\nLinks for {events[2].event_id[:8]}:")
    links = store.load_linked_events(events[2].event_id)
    for link_id in links:
        print(f"  -> {link_id[:8]}")
    
    # Cleanup
    os.unlink(db_path)
    print("\nTest completed!")
