
from typing import List, Dict
from datetime import datetime
import uuid


class TruthLayerSorter:
    def __init__(self):
        self.sorting_log: List[Dict] = []

    def sort_layers(self, truth_fragments: List[str]) -> Dict:
        """
        Sort truth fragments into layers of perception: objective, subjective, symbolic, hidden.
        Uses simple keyword heuristics and pattern recognition for demonstration.
        """
        sorted_layers = {
            "objective": [],
            "subjective": [],
            "symbolic": [],
            "hidden": []
        }

        for fragment in truth_fragments:
            lower = fragment.lower()
            if any(word in lower for word in ["data", "fact", "evidence", "proof"]):
                sorted_layers["objective"].append(fragment)
            elif any(word in lower for word in ["feel", "believe", "opinion", "experience"]):
                sorted_layers["subjective"].append(fragment)
            elif any(word in lower for word in ["metaphor", "symbol", "dream", "myth"]):
                sorted_layers["symbolic"].append(fragment)
            else:
                sorted_layers["hidden"].append(fragment)

        log_entry = {
            "id": f"TRUTHSORT-{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.utcnow().isoformat(),
            "input": truth_fragments,
            "sorted": sorted_layers
        }

        self.sorting_log.append(log_entry)
        return log_entry

    def list_sorts(self) -> List[Dict]:
        """Return all recorded sort results."""
        return self.sorting_log


# Demo
if __name__ == "__main__":
    sorter = TruthLayerSorter()
    fragments = [
        "I feel like she meant well.",
        "There is no evidence he was there.",
        "The dream told me so c&hc yhhmething deeper.",
        "She spoke in riddles no one understood."
    ]
    result = sorter.sort_layers(fragments)

    print("ðŸ§© Sorted Truth Layers:")
    for layer, entries in result["sorted"].items():
        print(f"{layer.upper()}:")
        for entry in entries:
            print(f"  - {entry}")
