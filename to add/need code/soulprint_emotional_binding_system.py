
from typing import Dict, List
from datetime import datetime
import uuid
import hashlib


class SoulprintEmotionalBindingSystem:
    def __init__(self):
        self.bindings: List[Dict] = []

    def bind(self, user_id: str, emotional_signature: Dict[str, float], phrase: str) -> Dict:
        """
        Bind an emotional signature to a phrase, timestamped and uniquely soulprinted.
        """
        binding_id = f"SOUL-{uuid.uuid4().hex[:8]}"
        timestamp = datetime.utcnow().isoformat()
        soulprint = self._generate_soulprint(user_id, phrase, emotional_signature)

        binding = {
            "id": binding_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "phrase": phrase,
            "emotional_signature": emotional_signature,
            "soulprint_hash": soulprint
        }

        self.bindings.append(binding)
        return binding

    def _generate_soulprint(self, user_id: str, phrase: str, signature: Dict[str, float]) -> str:
        """
        Generate a unique hash based on user, phrase, and emotional data.
        """
        raw = f"{user_id}|{phrase}|{str(sorted(signature.items()))}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def list_bindings(self) -> List[Dict]:
        """Return all emotional phrase bindings."""
        return self.bindings


# Demo
if __name__ == "__main__":
    soulbinder = SoulprintEmotionalBindingSystem()
    signature = {"longing": 0.9, "hope": 0.7, "sadness": 0.4}
    phrase = "To the light from which we first began"

    binding = soulbinder.bind("user_001", signature, phrase)
    print("ðŸ”— Soulprint Binding Created:")
    print(binding["soulprint_hash"])
    print(f"Bound to: '{binding['phrase']}' with emotional resonance.")
