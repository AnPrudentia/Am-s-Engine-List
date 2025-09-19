
from typing import Dict, Any
from datetime import datetime
import random
import uuid


class AmbientStateMapper:
    def __init__(self):
        self.state_log = []

    def map_environment(self, audio_level: float, light_level: float, temperature: float, movement_detected: bool) -> Dict[str, Any]:
        """
        Map ambient inputs to an emotional/environmental state snapshot.
        Inputs:
            audio_level: measured in decibels (dB)
            light_level: measured in lumens (lm)
            temperature: in Celsius
            movement_detected: boolean motion detection
        Returns:
            A dictionary with mapped emotional state and context snapshot.
        """
        emotional_state = self._interpret_state(audio_level, light_level, temperature, movement_detected)
        timestamp = datetime.utcnow().isoformat()
        snapshot_id = str(uuid.uuid4())

        snapshot = {
            "id": snapshot_id,
            "timestamp": timestamp,
            "inputs": {
                "audio_level": audio_level,
                "light_level": light_level,
                "temperature": temperature,
                "movement_detected": movement_detected
            },
            "emotional_state": emotional_state
        }

        self.state_log.append(snapshot)
        return snapshot

    def _interpret_state(self, audio, light, temp, motion) -> str:
        """
        Basic rule-based interpretation of ambient state.
        """
        if audio < 30 and light < 100 and not motion:
            return "serene"
        elif audio > 80 or (motion and audio > 60):
            return "tense"
        elif light > 800 and audio < 50:
            return "alert"
        elif temp > 28:
            return "restless"
        elif temp < 15:
            return "withdrawn"
        else:
            return random.choice(["neutral", "pensive", "focused"])

    def get_state_log(self):
        return self.state_log


# Demo
if __name__ == "__main__":
    mapper = AmbientStateMapper()

    readings = [
        (20, 50, 22, False),
        (85, 300, 23, True),
        (60, 900, 24, False),
        (10, 40, 14, False),
        (55, 400, 30, True)
    ]

    print("ðŸŒ Ambient State Mapping:")
    for audio, light, temp, move in readings:
        result = mapper.map_environment(audio, light, temp, move)
        print(f"{result['timestamp']} â†’ {result['emotional_state'].upper()}")
