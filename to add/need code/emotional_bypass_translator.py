from typing import Dict, List
from datetime import datetime
import uuid


class EmotionalBypassTranslator:
    def __init__(self):
        self.translations: List[Dict] = []

    def translate(self, bypass_behavior: str, observed_context: str = "") -> Dict:
        translation_id = f"EBT-{uuid.uuid4().hex[:8]}"
        root_emotion = self._infer_emotion(bypass_behavior)

        translation = {
            "id": translation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "behavior": bypass_behavior,
            "context": observed_context,
            "inferred_emotion": root_emotion
        }

        self.translations.append(translation)
        return translation

    def _infer_emotion(self, behavior: str) -> str:
        mapping = {
            "joking in serious moments": "fear of vulnerability",
            "changing subject": "emotional discomfort or avoidance",
            "over-intellectualizing": "desire for control over uncertainty",
            "dismissiveness": "suppressed shame or insecurity",
            "excessive positivity": "avoidance of grief or sadness"
        }
        return mapping.get(behavior.lower(), "emotionally ambiguous")

    def list_translations(self) -> List[Dict]:
        return self.translations
