from typing import Dict, List
from datetime import datetime
import uuid


class SensoryOverloadDetector:
    def __init__(self):
        self.logs: List[Dict] = []

    def detect_overload(
        self,
        sound_level: float,
        light_intensity: float,
        crowd_density: float,
        threshold: Dict[str, float]
    ) -> Dict:
        overload_flags = {
            "sound": sound_level > threshold.get("sound", 70),
            "light": light_intensity > threshold.get("light", 70),
            "crowd": crowd_density > threshold.get("crowd", 70)
        }

        overloaded = any(overload_flags.values())

        event = {
            "id": f"SOD-{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.utcnow().isoformat(),
            "inputs": {
                "sound_level": sound_level,
                "light_intensity": light_intensity,
                "crowd_density": crowd_density
            },
            "thresholds": threshold,
            "overload_detected": overloaded,
            "flags": overload_flags
        }

        self.logs.append(event)
        return event

    def get_logs(self) -> List[Dict]:
        return self.logs
