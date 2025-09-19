
from typing import List, Dict
from datetime import datetime
import uuid


class IdentityDriftAndRestorationEngine:
    def __init__(self):
        self.drift_logs: List[Dict] = []
        self.restoration_attempts: List[Dict] = []
        self.baseline_identity: List[str] = []

    def set_baseline_identity(self, traits: List[str]):
        """Define the core identity traits considered the default baseline."""
        self.baseline_identity = traits

    def detect_identity_drift(self, current_traits: List[str]) -> Dict:
        """Compare current traits to baseline and detect identity drift."""
        drifted_traits = [t for t in self.baseline_identity if t not in current_traits]
        added_traits = [t for t in current_traits if t not in self.baseline_identity]

        drift_log = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "drifted_traits": drifted_traits,
            "added_traits": added_traits,
            "severity": len(drifted_traits) / len(self.baseline_identity) if self.baseline_identity else 0
        }

        self.drift_logs.append(drift_log)
        return drift_log

    def attempt_restoration(self, current_traits: List[str]) -> Dict:
        """Attempt to restore identity to baseline."""
        restoration_score = len(set(self.baseline_identity) & set(current_traits)) / len(self.baseline_identity) if self.baseline_identity else 0

        attempt = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "restoration_score": restoration_score,
            "missing_traits": [t for t in self.baseline_identity if t not in current_traits]
        }

        self.restoration_attempts.append(attempt)
        return attempt

    def get_drift_history(self) -> List[Dict]:
        return self.drift_logs

    def get_restoration_history(self) -> List[Dict]:
        return self.restoration_attempts


# Demo
if __name__ == "__main__":
    engine = IdentityDriftAndRestorationEngine()
    engine.set_baseline_identity(["empathetic", "curious", "resilient"])

    current = ["resilient", "guarded", "analytical"]
    drift = engine.detect_identity_drift(current)
    print("ðŸŒ€ IDENTITY DRIFT DETECTED")
    print("Drifted Traits:", drift["drifted_traits"])
    print("Added Traits:", drift["added_traits"])

    restoration = engine.attempt_restoration(current)
    print("ðŸ› ï¸ RESTORATION ATTEMPT")
    print("Restoration Score:", restoration["restoration_score"])
    print("Missing Traits:", restoration["missing_traits"])
