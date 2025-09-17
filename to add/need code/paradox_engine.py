
from typing import List, Dict, Union
from datetime import datetime
import uuid


class ParadoxEngine:
    def __init__(self):
        self.paradox_log: List[Dict] = []

    def detect_paradox(self, statements: List[str]) -> Union[Dict, None]:
        """
        Detect logical paradoxes in a set of statements.
        Simple heuristic: contradictory pairs like A and not A.
        """
        normalized = [s.lower().strip().replace("not ", "Â¬") for s in statements]
        contradictions = []

        for i, stmt in enumerate(normalized):
            if "Â¬" + stmt.replace("Â¬", "") in normalized[i + 1:]:
                contradictions.append((stmt, "Â¬" + stmt.replace("Â¬", "")))

        if contradictions:
            paradox_id = f"PARADOX-{uuid.uuid4().hex[:8]}"
            entry = {
                "id": paradox_id,
                "timestamp": datetime.utcnow().isoformat(),
                "original_statements": statements,
                "contradictions": contradictions
            }
            self.paradox_log.append(entry)
            return entry
        return None

    def resolve_paradox(self, contradiction: tuple) -> str:
        """
        Offer a basic resolution: prioritize context or label as dual-truth.
        """
        stmt1, stmt2 = contradiction
        if "Â¬" in stmt1 or "Â¬" in stmt2:
            return f"ðŸŒ€ Resolution: Contextual dependency detected between '{stmt1}' and '{stmt2}' â€” dual-truth status granted."
        return f"âš–ï¸ Resolution: No contradiction confirmed under expanded semantics."

    def list_paradoxes(self) -> List[Dict]:
        """Return all detected paradoxes."""
        return self.paradox_log


# Demo
if __name__ == "__main__":
    pe = ParadoxEngine()

    statements = [
        "He is loyal",
        "He is not loyal",
        "The system is fair",
        "The system is broken"
    ]

    paradox = pe.detect_paradox(statements)
    if paradox:
        print("ðŸš¨ Paradox Detected:")
        print(paradox)

        for pair in paradox["contradictions"]:
            print(pe.resolve_paradox(pair))
    else:
        print("âœ… No paradox detected.")
