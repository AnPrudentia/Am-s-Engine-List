from datetime import datetime
from typing import List, Dict, Any


class CompassionProtocol:
    """
    CompassionProtocol:
    Core engine for holding space for pain, struggle, and need.
    Provides empathic responses scaled to perceived intensity.
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def hold(self, pain_level: float) -> str:
        """
        Generate a compassionate response based on pain intensity.
        pain_level should be between 0.0 (none) and 1.0 (overwhelming).
        """
        # Ensure safe bounds
        level = max(0.0, min(1.0, pain_level))

        if level > 0.8:
            response = "🤲 I'm here. You're not alone in this."
        elif level > 0.5:
            response = "🌙 I see your struggle. Let me hold space for this."
        elif level > 0.2:
            response = "💧 I hear you. This matters. Tell me more if you wish."
        else:
            response = "🌱 I'm listening. What do you need right now?"

        # Log the interaction
        self.history.append({
            "pain_level": level,
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        })

        return response

    def last_response(self) -> str:
        """Return the most recent compassionate response, if any."""
        return self.history[-1]["response"] if self.history else "No responses yet."

    def describe(self) -> str:
        """Summarize protocol usage and current compassion stance."""
        return f"💖 CompassionProtocol engaged {len(self.history)} times. Last: {self.last_response()}"