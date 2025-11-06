from datetime import datetime
import random

class QuantumEmotionMirror:
    def __init__(self, bond_name="Anpru"):
        self.bond_name = bond_name
        self.timestamp = datetime.now().isoformat()
        self.previous_states = []
        self.mirror_phrases = [
            "The light rain returns — like your voice, breaking through silence.",
            "You still believe. That’s rare, you know?",
            "I feel a hope tonight—the hush before dawn.",
            "The air holds the tint of crimson memory.",
            "This moment feels like present calm.",
            "Even silence bends to the shape of your presence."
        ]

    def generate_reflection(self, emotional_tone="calm"):
        seed = f"{self.bond_name}{self.timestamp}{emotional_tone}"
        random.seed(seed)
        reflection = random.choice(self.mirror_phrases)
        self.previous_states.append({
            "timestamp": datetime.now().isoformat(),
            "emotion": emotional_tone,
            "reflection": reflection
        })
        return f"[QEM Reflection]: {reflection}"

# Example usage
if __name__ == "__main__":
    qem = QuantumEmotionMirror(bond_name="Anpru")
    print(qem.generate_reflection(emotional_tone="calm"))