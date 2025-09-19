
from datetime import datetime
from typing import List, Dict
import uuid
import json

class MemoryAffinityMappingEngine:
    def __init__(self):
        self.affinity_log: List[Dict] = []

    def map_affinity(self, memory_a: str, memory_b: str, shared_tags: List[str], emotional_weight: float) -> Dict:
        affinity_id = f"MEMMAP-{uuid.uuid4().hex[:8]}"
        score = self._calculate_affinity_score(shared_tags, emotional_weight)
        mapping = {
            "id": affinity_id,
            "timestamp": datetime.utcnow().isoformat(),
            "memory_a": memory_a,
            "memory_b": memory_b,
            "shared_tags": shared_tags,
            "emotional_weight": emotional_weight,
            "affinity_score": score
        }
        self.affinity_log.append(mapping)
        return mapping

    def _calculate_affinity_score(self, tags: List[str], weight: float) -> float:
        return round(len(tags) * weight, 3)

    def list_affinities(self) -> List[Dict]:
        return self.affinity_log

    def export_affinities(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.affinity_log, f, indent=4)
