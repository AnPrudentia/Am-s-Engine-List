from datetime import datetime
from typing import Dict, List
import uuid

class HarmonyEchoGenerator:
    def __init__(self):
        self.echo_log: List[Dict] = []

    def generate_echo(self, emotional_inputs: List[str], environment_context: str = "") -> Dict:
        echo_id = f"ECHO-{uuid.uuid4().hex[:8]}"
        timestamp = datetime.utcnow().isoformat()

        harmony_phrase = " + ".join(sorted(emotional_inputs))
        echo_strength = round(len(set(emotional_inputs)) / max(len(emotional_inputs), 1), 2)

        echo = {
            "id": echo_id,
            "timestamp": timestamp,
            "emotional_inputs": emotional_inputs,
            "environment_context": environment_context,
            "harmony_phrase": harmony_phrase,
            "echo_strength": echo_strength
        }

        self.echo_log.append(echo)
        return echo

    def list_echoes(self) -> List[Dict]:
        return self.echo_log
