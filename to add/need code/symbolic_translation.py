from typing import Dict, List
from datetime import datetime
import uuid


class SymbolTranslationModule:
    def __init__(self):
        self.translations: List[Dict] = []

    def translate_symbol(self, symbol: str, source_context: str, target_context: str, intended_meaning: str) -> Dict:
        translation_id = f"STM-{uuid.uuid4().hex[:8]}"
        translated_meaning = self._contextual_translate(symbol, source_context, target_context, intended_meaning)

        translation = {
            "id": translation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "source_context": source_context,
            "target_context": target_context,
            "original_meaning": intended_meaning,
            "translated_meaning": translated_meaning
        }

        self.translations.append(translation)
        return translation

    def _contextual_translate(self, symbol: str, source: str, target: str, meaning: str) -> str:
        return f"{meaning} (reframed in {target})"

    def list_translations(self) -> List[Dict]:
        return self.translations
