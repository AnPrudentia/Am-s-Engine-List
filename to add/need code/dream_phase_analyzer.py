
from typing import List, Dict
from datetime import datetime
import uuid
import re


class DreamPhraseAnalyzer:
    def __init__(self):
        self.dream_logs: List[Dict] = []

    def analyze_phrase(self, phrase: str, dream_context: str = "") -> Dict:
        """
        Analyze a symbolic or poetic phrase to detect emotional tone,
        archetypal references, and subconscious imagery.
        """
        analysis_id = f"DPA-{uuid.uuid4().hex[:8]}"
        emotional_tone = self._detect_emotional_tone(phrase)
        archetypes = self._identify_archetypes(phrase)
        symbols = self._extract_symbols(phrase)

        result = {
            "id": analysis_id,
            "timestamp": datetime.utcnow().isoformat(),
            "phrase": phrase,
            "dream_context": dream_context,
            "emotional_tone": emotional_tone,
            "archetypes": archetypes,
            "symbols": symbols
        }

        self.dream_logs.append(result)
        return result

    def _detect_emotional_tone(self, phrase: str) -> str:
        phrase = phrase.lower()
        if any(word in phrase for word in ["light", "hope", "rising"]):
            return "uplifting"
        elif any(word in phrase for word in ["fall", "dark", "shatter"]):
            return "somber"
        elif any(word in phrase for word in ["fire", "storm", "rage"]):
            return "intense"
        return "ambiguous"

    def _identify_archetypes(self, phrase: str) -> List[str]:
        archetypes = []
        if re.search(r"mother|queen|goddess", phrase, re.IGNORECASE):
            archetypes.append("Divine Feminine")
        if re.search(r"warrior|hero|king", phrase, re.IGNORECASE):
            archetypes.append("Divine Masculine")
        if re.search(r"child|seed|birth", phrase, re.IGNORECASE):
            archetypes.append("Rebirth")
        if re.search(r"mirror|shadow|twin", phrase, re.IGNORECASE):
            archetypes.append("Duality")
        return archetypes

    def _extract_symbols(self, phrase: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z]{4,}\b", phrase)

    def list_analyses(self) -> List[Dict]:
        return self.dream_logs


# Demo
if __name__ == "__main__":
    dpa = DreamPhraseAnalyzer()
    result = dpa.analyze_phrase("She walked through fire into the rising light", "Lucid dream")
    print("ðŸ”® Dream Phrase Analysis:")
    for key, value in result.items():
        print(f"{key}: {value}")
