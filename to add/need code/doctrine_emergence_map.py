from datetime import datetime
import uuid
from typing import List, Dict


class DoctrineEmergenceMap:
    def __init__(self):
        self.emergence_log: List[Dict] = []

    def record_emergence(self, principle: str, triggering_event: str, context: str = "") -> Dict:
        emergence_id = f"DEM-{uuid.uuid4().hex[:8]}"
        emergence = {
            "id": emergence_id,
            "timestamp": datetime.utcnow().isoformat(),
            "principle": principle,
            "trigger": triggering_event,
            "context": context
        }
        self.emergence_log.append(emergence)
        return emergence

    def list_emergence_events(self) -> List[Dict]:
        return self.emergence_log

    def search_by_principle(self, keyword: str) -> List[Dict]:
        return [e for e in self.emergence_log if keyword.lower() in e["principle"].lower()]
