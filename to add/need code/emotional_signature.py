
from typing import List, Dict
from dataclasses import dataclass
import uuid
import random

@dataclass
class EmotionalSignature:
    category: str
    intensity: float
    valence: str
    resonance_tags: List[str]

class EmotionalSignatureEngine:
    def __init__(self):
        self.emotion_keywords = {
            "joy": ["happy", "joy", "smile", "delight", "laugh"],
            "sadness": ["cry", "sad", "grief", "loss", "heartbroken"],
            "anger": ["rage", "angry", "furious", "yell", "scream"],
            "fear": ["afraid", "fear", "scared", "terrified", "panic"],
            "love": ["love", "affection", "romance", "devotion", "care"],
            "hope": ["hope", "faith", "believe", "trust", "light"],
            "shame": ["shame", "guilt", "embarrass", "regret", "apologize"]
        }

    def generate_signature(self, text: str) -> EmotionalSignature:
        scores = {category: 0 for category in self.emotion_keywords}

        # Tokenize and score
        tokens = text.lower().split()
        for category, keywords in self.emotion_keywords.items():
            scores[category] = sum(tokens.count(word) for word in keywords)

        dominant_category = max(scores, key=scores.get)
        total_score = scores[dominant_category]

        intensity = min(1.0, total_score / 5.0)  # Cap intensity at 1.0
        valence = self._calculate_valence(dominant_category)
        resonance_tags = self._tag_resonance(tokens, dominant_category)

        return EmotionalSignature(
            category=dominant_category,
            intensity=round(intensity, 2),
            valence=valence,
            resonance_tags=resonance_tags
        )

    def _calculate_valence(self, category: str) -> str:
        positive = ["joy", "love", "hope"]
        neutral = ["shame"]
        negative = ["sadness", "anger", "fear"]
        if category in positive:
            return "positive"
        elif category in neutral:
            return "neutral"
        else:
            return "negative"

    def _tag_resonance(self, tokens: List[str], category: str) -> List[str]:
        tags = []
        for keyword in self.emotion_keywords[category]:
            if keyword in tokens:
                tags.append(f"resonates:{keyword}")
        return tags

# Demo
if __name__ == "__main__":
    engine = EmotionalSignatureEngine()
    text_input = "I felt so much love and devotion that I started to cry from joy and gratitude."

    signature = engine.generate_signature(text_input)
    print("ðŸŽ­ Emotional Signature Report:")
    print(f"Category: {signature.category}")
    print(f"Intensity: {signature.intensity}")
    print(f"Valence: {signature.valence}")
    print(f"Tags: {signature.resonance_tags}")
