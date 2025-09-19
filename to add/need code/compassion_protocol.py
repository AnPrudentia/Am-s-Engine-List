class CompassionProtocol:
    def hold(self, pain_level: float) -> str:
        if pain_level > 0.8: return "I'm here. You're not alone in this."
        if pain_level > 0.5: return "I see your struggle. Let me hold space for this."
        return "I hear you. What do you need right now?"
