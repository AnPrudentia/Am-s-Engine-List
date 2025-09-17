
from datetime import datetime
import uuid
from typing import List, Dict


class EmotionalCadenceMapper:
    def __init__(self):
        self.cadence_log: List[Dict] = []

    def analyze_cadence(self, emotional_inputs: List[Dict[str, float]]) -> Dict:
        timestamp = datetime.utcnow().isoformat()
        cadence_profile = {}
        shift_sequence = []

        for i in range(1, len(emotional_inputs)):
            prev = emotional_inputs[i - 1]
            curr = emotional_inputs[i]
            for emotion in curr:
                shift = curr[emotion] - prev.get(emotion, 0.0)
                shift_sequence.append((emotion, round(shift, 3)))

        for emotion, shift in shift_sequence:
            if emotion not in cadence_profile:
                cadence_profile[emotion] = {"rise": 0, "fall": 0}
            if shift > 0:
                cadence_profile[emotion]["rise"] += 1
            elif shift < 0:
                cadence_profile[emotion]["fall"] += 1

        result = {
            "id": f"CADENCE-{uuid.uuid4().hex[:8]}",
            "timestamp": timestamp,
            "profile": cadence_profile,
            "shifts": shift_sequence
        }
        self.cadence_log.append(result)
        return result

    def get_cadence_log(self) -> List[Dict]:
        return self.cadence_log


# Demo
if __name__ == "__main__":
    mapper = EmotionalCadenceMapper()
    emotion_series = [
        {"joy": 0.3, "sadness": 0.1, "anger": 0.0},
        {"joy": 0.5, "sadness": 0.2, "anger": 0.0},
        {"joy": 0.2, "sadness": 0.4, "anger": 0.1},
        {"joy": 0.3, "sadness": 0.3, "anger": 0.2}
    ]

    cadence = mapper.analyze_cadence(emotion_series)
    print("ðŸ§­ Emotional Cadence Map:")
    print(cadence)