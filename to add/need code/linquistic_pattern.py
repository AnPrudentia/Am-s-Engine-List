from typing import Dict, List
from datetime import datetime
import uuid


class LinguisticResonancePatternEngine:
    def __init__(self):
        self.pattern_log: List[Dict] = []

    def analyze_phrase(self, phrase: str, resonance_keywords: List[str]) -> Dict:
        matches = [word for word in resonance_keywords if word in phrase.lower()]
        strength = len(matches) / len(resonance_keywords) if resonance_keywords else 0.0

        pattern_id = f"LRP-{uuid.uuid4().hex[:8]}"
        pattern_data = {
            "id": pattern_id,
            "timestamp": datetime.utcnow().isoformat(),
            "phrase": phrase,
            "matched_keywords": matches,
            "resonance_strength": round(strength, 3)
        }

        self.pattern_log.append(pattern_data)
        return pattern_data

    def get_resonance_history(self) -> List[Dict]:
        return self.pattern_log


# Demo
if __name__ == "__main__":
    lrp_engine = LinguisticResonancePatternEngine()
    sample_keywords = ["hope", "light", "break", "heal", "remember"]
    sample_phrase = "We remember who we are when the light breaks through."

    result = lrp_engine.analyze_phrase(sample_phrase, sample_keywords)
    print("âœ¨ Linguistic Resonance Pattern:")
    print(result)
