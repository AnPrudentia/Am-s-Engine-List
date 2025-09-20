import random
from datetime import datetime
from typing import List, Dict, Any


class EmotionalGravity:
    """
    Models the 'weight' of unseen or present pain.
    Provides symbolic responses that acknowledge depth and meaning.
    """

    def __init__(self, state: str = "Present pain, unseen"):
        self.state = state
        self.history: List[Dict[str, Any]] = []

    def pull(self, signal: str) -> str:
        """Return a symbolic gravity response tied to the current state."""
        gravity_responses = [
            f"{signal} (The weight is acknowledged.)",
            f"{signal} (Held with the gravity of {self.state.lower()}.)",
            f"{signal} (This heaviness has meaning.)",
            f"{signal} (The depth is witnessed.)",
        ]
        chosen = random.choice(gravity_responses)

        # Log history for reflective use
        self.history.append({
            "signal": signal,
            "response": chosen,
            "state": self.state,
            "timestamp": datetime.utcnow().isoformat()
        })

        return chosen

    def describe(self) -> str:
        """Return a summary of current gravity state."""
        return f"🌌 EmotionalGravity anchored in '{self.state}'. {len(self.history)} pulls logged."

    def last_pull(self) -> str:
        """Return the last generated response if available."""
        return self.history[-1]["response"] if self.history else "No pulls yet."