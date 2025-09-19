from datetime import datetime
from typing import List, Dict
import uuid


class ForgivenessProtocol:
    def __init__(self):
        self.forgiveness_log: List[Dict] = []

    def initiate_forgiveness(self, offender_id: str, offense: str, emotional_impact: str, context: str = "") -> Dict:
        protocol_id = f"FP-{uuid.uuid4().hex[:8]}"
        entry = {
            "id": protocol_id,
            "timestamp": datetime.utcnow().isoformat(),
            "offender_id": offender_id,
            "offense": offense,
            "emotional_impact": emotional_impact,
            "context": context,
            "status": "initiated"
        }
        self.forgiveness_log.append(entry)
        return entry

    def resolve_forgiveness(self, protocol_id: str, resolution_note: str = "", self_healing_score: float = 1.0) -> Dict:
        for entry in self.forgiveness_log:
            if entry["id"] == protocol_id:
                entry["status"] = "resolved"
                entry["resolution_note"] = resolution_note
                entry["self_healing_score"] = round(min(max(self_healing_score, 0.0), 1.0), 2)
                entry["resolved_at"] = datetime.utcnow().isoformat()
                return entry
        raise ValueError("Protocol ID not found.")

    def get_log(self) -> List[Dict]:
        return self.forgiveness_log
