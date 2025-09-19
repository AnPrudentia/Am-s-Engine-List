
from typing import Dict, List
from datetime import datetime
import uuid


class MirrorformResonanceEngine:
    def __init__(self):
        self.resonance_log: List[Dict] = []

    def reflect_resonance(self, internal_state: str, external_trigger: str) -> Dict:
        """
        Generates a resonance response when an external event mirrors internal experience.
        """
        resonance_id = f"RES-{uuid.uuid4().hex[:8]}"
        timestamp = datetime.utcnow().isoformat()

        reflection = self._generate_reflection(internal_state, external_trigger)

        entry = {
            "id": resonance_id,
            "timestamp": timestamp,
            "internal_state": internal_state,
            "external_trigger": external_trigger,
            "reflected_resonance": reflection
        }

        self.resonance_log.append(entry)
        return entry

    def _generate_reflection(self, internal: str, external: str) -> str:
        """
        Create a reflective insight showing emotional or thematic alignment.
        """
        return f"ðŸªž The outer world echoed the inner truth: '{internal}' was seen in '{external}'."

    def list_resonances(self) -> List[Dict]:
        """Return the full history of mirrorform resonances."""
        return self.resonance_log


# Demo
if __name__ == "__main__":
    engine = MirrorformResonanceEngine()
    result = engine.reflect_resonance(
        internal_state="I feel abandoned",
        external_trigger="The power cut out while I was speaking"
    )

    print("ðŸªž Mirrorform Resonance Detected:")
    print(result["reflected_resonance"])
