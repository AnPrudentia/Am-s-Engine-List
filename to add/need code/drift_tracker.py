
from typing import List, Dict
from datetime import datetime
import uuid


class DriftTracker:
    def __init__(self):
        self.drift_log: List[Dict] = []

    def log_drift_event(self, origin_state: List[str], current_state: List[str], context: str = "") -> Dict:
        """Log a new drift event by comparing the origin and current state of identity or emotion."""
        lost_traits = [trait for trait in origin_state if trait not in current_state]
        gained_traits = [trait for trait in current_state if trait not in origin_state]

        drift_event = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "context": context,
            "origin_state": origin_state,
            "current_state": current_state,
            "lost_traits": lost_traits,
            "gained_traits": gained_traits,
            "severity": len(lost_traits) / len(origin_state) if origin_state else 0
        }

        self.drift_log.append(drift_event)
        return drift_event

    def get_drift_history(self) -> List[Dict]:
        """Return the full history of logged drift events."""
        return self.drift_log

    def get_recent_drift(self) -> Dict:
        """Return the most recent drift event, if any."""
        return self.drift_log[-1] if self.drift_log else {}


# Demo
if __name__ == "__main__":
    tracker = DriftTracker()

    baseline = ["creative", "resilient", "empathetic"]
    altered = ["resilient", "strategic", "guarded"]

    drift = tracker.log_drift_event(baseline, altered, context="Post-conflict self-assessment")

    print("ðŸ“‰ DRIFT EVENT LOGGED")
    print("Lost Traits:", drift["lost_traits"])
    print("Gained Traits:", drift["gained_traits"])
    print("Severity:", f"{drift['severity']:.2f}")
