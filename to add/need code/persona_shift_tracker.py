from typing import Dict, List
from datetime import datetime
import uuid


class PersonaShiftTracker:
    def __init__(self):
        self.shifts: List[Dict] = []

    def log_shift(self, previous_persona: str, new_persona: str, catalyst_event: str, observed_change: str = "") -> Dict:
        shift_id = f"PST-{uuid.uuid4().hex[:8]}"
        shift_record = {
            "id": shift_id,
            "timestamp": datetime.utcnow().isoformat(),
            "previous_persona": previous_persona,
            "new_persona": new_persona,
            "catalyst_event": catalyst_event,
            "observed_change": observed_change
        }
        self.shifts.append(shift_record)
        return shift_record

    def get_shifts_by_persona(self, persona_name: str) -> List[Dict]:
        return [s for s in self.shifts if s["previous_persona"] == persona_name or s["new_persona"] == persona_name]

    def list_all_shifts(self) -> List[Dict]:
        return self.shifts
