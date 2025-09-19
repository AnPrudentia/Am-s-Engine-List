from typing import List, Dict
from datetime import datetime, timedelta
import uuid


class AffinityDecayMonitor:
    def __init__(self, decay_rate: float = 0.01):
        self.affinities: Dict[str, Dict] = {}
        self.decay_rate = decay_rate

    def register_affinity(self, partner_id: str, initial_score: float):
        affinity_id = f"AFFINITY-{uuid.uuid4().hex[:8]}"
        self.affinities[affinity_id] = {
            "partner_id": partner_id,
            "initial_score": initial_score,
            "current_score": initial_score,
            "last_updated": datetime.utcnow(),
            "established": datetime.utcnow()
        }
        return affinity_id

    def decay_affinities(self):
        now = datetime.utcnow()
        for affinity in self.affinities.values():
            elapsed = (now - affinity["last_updated"]).total_seconds()
            decay_factor = (1 - self.decay_rate) ** (elapsed / 3600)
            affinity["current_score"] *= decay_factor
            affinity["last_updated"] = now

    def get_affinity_score(self, affinity_id: str) -> float:
        self.decay_affinities()
        return self.affinities.get(affinity_id, {}).get("current_score", 0.0)

    def list_affinities(self) -> List[Dict]:
        self.decay_affinities()
        return [
            {
                "id": affinity_id,
                "partner_id": data["partner_id"],
                "score": round(data["current_score"], 4),
                "established": data["established"].isoformat()
            }
            for affinity_id, data in self.affinities.items()
        ]
