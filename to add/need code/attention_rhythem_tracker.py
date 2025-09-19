from typing import Dict, List
from datetime import datetime
import uuid


class AttentionRhythmTracker:
    def __init__(self):
        self.sessions: List[Dict] = []

    def log_attention(self, timestamp: str, focus_level: int, context: str = "") -> Dict:
        session_id = f"ART-{uuid.uuid4().hex[:8]}"
        session = {
            "id": session_id,
            "timestamp": timestamp,
            "focus_level": focus_level,
            "context": context
        }
        self.sessions.append(session)
        return session

    def get_focus_trend(self) -> Dict:
        if not self.sessions:
            return {"average_focus": 0, "peak": None, "dip": None}

        levels = [s["focus_level"] for s in self.sessions]
        average = sum(levels) / len(levels)
        peak = max(self.sessions, key=lambda s: s["focus_level"])
        dip = min(self.sessions, key=lambda s: s["focus_level"])

        return {
            "average_focus": round(average, 2),
            "peak": peak,
            "dip": dip
        }

    def list_sessions(self) -> List[Dict]:
        return self.sessions
