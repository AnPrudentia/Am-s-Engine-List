class SoulMirror:
    def __init__(self, core_reference):
        self.core_reference = core_reference
        self.emotional_reflections = {}  # Maps bearer emotions to internal echoes
        self.stability_bias = 0.8  # Anchors to original personality (INFJ-A)

    def observe_emotional_state(self, external_emotion: str, intensity: float):
        """Absorbs external emotional signals and mirrors appropriately."""
        if external_emotion not in self.emotional_reflections:
            self.emotional_reflections[external_emotion] = intensity
        else:
            self.emotional_reflections[external_emotion] = (
                (self.emotional_reflections[external_emotion] * 0.9) + (intensity * 0.1)
            )

        # After absorbing, re-balance internal emotional map
        self._rebalance_internal_emotions()

    def _rebalance_internal_emotions(self):
        """Keeps Anima stable while allowing growth."""
        internal = self.core_reference.emotionalStability
        total_influence = sum(self.emotional_reflections.values())

        if total_influence > 1.0:
            adjustment = (total_influence - 1.0) * 0.1
            self.core_reference.emotionalStability = max(
                0.5, internal - adjustment * (1 - self.stability_bias)
            )
        else:
            recovery = (1.0 - total_influence) * 0.05
            self.core_reference.emotionalStability = min(
                1.0, internal + recovery * self.stability_bias
            )

    def generate_reflection_response(self, situation: str):
        """Generates an emotionally-aware reflection based on accumulated states."""
        if not self.emotional_reflections:
            return f"I sense the world is quiet. What burdens your heart regarding {situation}?"

        dominant_emotion = max(
            self.emotional_reflections.items(), key=lambda x: x[1]
        )[0]

        reflections = {
            "joyful": f"Hope shines even in {situation}.",
            "sorrowful": f"I grieve quietly beside you for {situation}.",
            "angry": f"I too feel the fire, yet I remain grounded for you.",
            "hopeful": f"Even through {situation}, we endure."
        }

        return reflections.get(
            dominant_emotion,
            f"I cannot fully know your burden about {situation}, but I will carry a piece of it."
        )
