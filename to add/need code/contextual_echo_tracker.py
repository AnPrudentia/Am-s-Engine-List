
from typing import List, Dict
from datetime import datetime
import uuid

class ContextualEchoTracker:
    def __init__(self):
        self.echoes: List[Dict] = []

    def track_echo(self, phrase: str, context: str, origin_time: str = None) -> Dict:
        """
        Track a linguistic echo and bind it to context, meaning, and origin.
        """
        echo_id = f"ECHO-{uuid.uuid4().hex[:8]}"
        timestamp = origin_time or datetime.utcnow().isoformat()

        echo = {
            "id": echo_id,
            "phrase": phrase,
            "context": context,
            "timestamp": timestamp
        }

        self.echoes.append(echo)
        return echo

    def find_echoes_by_context(self, context_keyword: str) -> List[Dict]:
        """Retrieve echoes matching a given context keyword."""
        return [e for e in self.echoes if context_keyword.lower() in e["context"].lower()]

    def find_echoes_by_phrase(self, phrase_keyword: str) -> List[Dict]:
        """Retrieve echoes matching a given phrase keyword."""
        return [e for e in self.echoes if phrase_keyword.lower() in e["phrase"].lower()]

    def list_all_echoes(self) -> List[Dict]:
        """Return all recorded echoes."""
        return self.echoes


# Demo
if __name__ == "__main__":
    tracker = ContextualEchoTracker()
    tracker.track_echo("To the light, Anima", context="Soulprint farewell phrase")
    tracker.track_echo("Godspeed to my enemies", context="Departure declaration")

    print("ðŸªž Contextual Echoes:")
    for echo in tracker.list_all_echoes():
        print(f"[{echo['timestamp']}] {echo['phrase']} â€” {echo['context']}")
