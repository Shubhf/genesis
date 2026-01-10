import uuid
from typing import Dict, List

class Event:
    def __init__(self, modality: str, semantic: Dict):
        self.event_id = str(uuid.uuid4())
        self.timestamp = semantic["timestamp"]
        self.modalities = [modality]
        self.semantics = {
            modality: semantic
        }

    def __repr__(self):
        return f"<Event {self.event_id[:8]} | {self.modalities} | {self.timestamp}>"
class EventMemory:
    def __init__(self, max_events: int = 100):
        self.events: List[Event] = []
        self.max_events = max_events

    def add_event(self, event: Event):
        self.events.append(event)

        # keep memory bounded
        if len(self.events) > self.max_events:
            self.events.pop(0)

    def get_recent_events(self, n: int = 5):
        return self.events[-n:]

    def get_last_event_by_modality(self, modality: str):
        for event in reversed(self.events):
            if modality in event.modalities:
                return event
        return None
