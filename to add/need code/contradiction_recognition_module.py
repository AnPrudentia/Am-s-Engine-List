
from typing import List, Dict, Tuple
from datetime import datetime
import uuid


class ContradictionRecognitionModule:
    def __init__(self):
        self.contradiction_history: List[Dict] = []

    def recognize(self, assertions: List[str]) -> List[Dict]:
        """
        Identify direct contradictions between assertions.
        Uses simple pattern matching for 'X' vs 'not X' and tracks historical context.
        """
        contradictions = []
        normalized_map = {a.lower().replace("not ", "Â¬"): a for a in assertions}

        checked = set()
        for norm_a, raw_a in normalized_map.items():
            inverse = "Â¬" + norm_a.replace("Â¬", "")
            if inverse in normalized_map and (norm_a, inverse) not in checked:
                contradiction = {
                    "id": f"CONTRA-{uuid.uuid4().hex[:8]}",
                    "timestamp": datetime.utcnow().isoformat(),
                    "statement_1": raw_a,
                    "statement_2": normalized_map[inverse]
                }
                contradictions.append(contradiction)
                self.contradiction_history.append(contradiction)
                checked.add((norm_a, inverse))
                checked.add((inverse, norm_a))

        return contradictions

    def get_history(self) -> List[Dict]:
        """Retrieve the history of all recognized contradictions."""
        return self.contradiction_history


# Demo
if __name__ == "__main__":
    crm = ContradictionRecognitionModule()
    test_assertions = [
        "She is honest",
        "She is not honest",
        "The machine is operational",
        "The machine is not operational"
    ]

    found = crm.recognize(test_assertions)
    if found:
        print("âš ï¸ Contradictions Found:")
        for c in found:
            print(f"{c['statement_1']} âŸ· {c['statement_2']} (ID: {c['id']})")
    else:
        print("âœ… No contradictions detected.")
