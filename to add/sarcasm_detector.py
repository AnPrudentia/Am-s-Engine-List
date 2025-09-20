
from typing import Dict
import uuid
from datetime import datetime

class SarcasmDetector:
    def __init__(self):
        self.detection_log = []

    def detect(self, text: str, emotional_tone: Dict[str, float], context_clarity: float = 0.5) -> Dict[str, float]:
        """
        Detect sarcasm based on tonal contradiction and lack of contextual clarity.
        Returns a sarcasm probability score with contributing factors.
        """
        contradiction_score = self._evaluate_tonal_contradiction(emotional_tone)
        ambiguity_factor = 1.0 - context_clarity
        sarcasm_score = min(1.0, round((contradiction_score + ambiguity_factor) / 2, 3))

        result = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "text": text,
            "contradiction_score": contradiction_score,
            "ambiguity_factor": ambiguity_factor,
            "sarcasm_score": sarcasm_score
        }

        self.detection_log.append(result)
        return result

    def _evaluate_tonal_contradiction(self, tone: Dict[str, float]) -> float:
        """
        Calculate tonal contradiction: e.g., high joy + high contempt = sarcasm
        """
        joy = tone.get("joy", 0.0)
        contempt = tone.get("contempt", 0.0)
        annoyance = tone.get("annoyance", 0.0)
        contradiction = abs(joy - max(contempt, annoyance))
        return round(min(1.0, contradiction), 3)

    def get_log(self):
        return self.detection_log

# Demo
if __name__ == "__main__":
    detector = SarcasmDetector()
    examples = [
        {
            "text": "Oh wow, I *totally* believe you.",
            "tone": {"joy": 0.7, "contempt": 0.8},
            "clarity": 0.3
        },
        {
            "text": "What a fantastic idea... said no one ever.",
            "tone": {"joy": 0.2, "annoyance": 0.6},
            "clarity": 0.4
        }
    ]

    print("ðŸ§ª Sarcasm Detection:")
    for ex in examples:
        result = detector.detect(ex["text"], ex["tone"], ex["clarity"])
        print(f"- "{ex['text']}" â†’ Sarcasm: {result['sarcasm_score']}")
