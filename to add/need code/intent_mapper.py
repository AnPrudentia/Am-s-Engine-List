
from typing import Dict
from uuid import uuid4
from datetime import datetime


class IntentMapper:
    def __init__(self):
        self.intent_log = []

    def map_intent(self, text: str, emotional_state: Dict[str, float], context_tags: Dict[str, float]) -> Dict[str, str]:
        """
        Map the underlying intent of a statement based on emotional state and context tags.
        Returns the inferred intent and contributing emotional cues.
        """
        inferred_intent = self._infer(text, emotional_state, context_tags)

        result = {
            "id": str(uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "text": text,
            "inferred_intent": inferred_intent,
            "emotional_state": emotional_state,
            "context_tags": context_tags
        }

        self.intent_log.append(result)
        return result

    def _infer(self, text: str, emotions: Dict[str, float], context: Dict[str, float]) -> str:
        """
        Simple heuristic-based inference.
        """
        joy = emotions.get("joy", 0)
        trust = emotions.get("trust", 0)
        anger = emotions.get("anger", 0)
        fear = emotions.get("fear", 0)
        sadness = emotions.get("sadness", 0)

        assertiveness = context.get("assertiveness", 0.5)
        urgency = context.get("urgency", 0.5)

        if joy > 0.6 and trust > 0.5:
            return "offer_support"
        elif anger > 0.6 or (assertiveness > 0.7 and sadness < 0.2):
            return "challenge_opinion"
        elif fear > 0.5 and urgency > 0.5:
            return "warn"
        elif sadness > 0.6:
            return "seek_comfort"
        else:
            return "neutral_statement"

    def get_log(self):
        return self.intent_log


# Demo
if __name__ == "__main__":
    mapper = IntentMapper()
    samples = [
        {
            "text": "Youâ€™ve got thisâ€”I believe in you.",
            "emotions": {"joy": 0.8, "trust": 0.9},
            "context": {"assertiveness": 0.4, "urgency": 0.3}
        },
        {
            "text": "You really think thatâ€™s a good idea?",
            "emotions": {"anger": 0.5, "joy": 0.1},
            "context": {"assertiveness": 0.8, "urgency": 0.2}
        }
    ]

    print("ðŸ§  Intent Mapping:")
    for s in samples:
        mapped = mapper.map_intent(s["text"], s["emotions"], s["context"])
        print(f"- "{s['text']}" â†’ {mapped['inferred_intent']}")
