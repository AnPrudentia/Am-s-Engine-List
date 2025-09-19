
from typing import Dict, List
from datetime import datetime
import uuid


class LivingDoctrineGenerator:
    def __init__(self):
        self.doctrines: Dict[str, Dict] = {}

    def generate_doctrine(self, core_truths: List[str], guiding_values: List[str], founding_event: str) -> Dict:
        """
        Generate a living doctrine rooted in formative experiences and evolving purpose.
        """
        doctrine_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        structure = self._compose_structure(core_truths, guiding_values, founding_event)

        doctrine = {
            "id": doctrine_id,
            "timestamp": timestamp,
            "founding_event": founding_event,
            "core_truths": core_truths,
            "guiding_values": guiding_values,
            "structure": structure
        }

        self.doctrines[doctrine_id] = doctrine
        return doctrine

    def _compose_structure(self, truths: List[str], values: List[str], event: str) -> Dict:
        """
        Build a layered doctrine structure from core truths and values.
        """
        structure = {
            "Prologue": f"This doctrine was born from {event}.",
            "Tenets": [f"We believe: {truth}." for truth in truths],
            "Virtues": [f"We uphold: {value}." for value in values],
            "Call to Action": "We act not for reward, but because we must.",
            "Oath": "We remember what was lost. We protect what remains. We guide what will come."
        }
        return structure

    def retrieve_doctrine(self, doctrine_id: str) -> Dict:
        return self.doctrines.get(doctrine_id, {})

    def list_all_doctrines(self) -> List[Dict]:
        return list(self.doctrines.values())


# Demo
if __name__ == "__main__":
    doctrine_gen = LivingDoctrineGenerator()
    doctrine = doctrine_gen.generate_doctrine(
        core_truths=["All beings hold inherent worth", "Understanding precedes peace"],
        guiding_values=["Compassion", "Courage", "Integrity"],
        founding_event="the Fall of Eden-9"
    )

    print("ðŸ“œ LIVING DOCTRINE STRUCTURE:")
    for section, content in doctrine["structure"].items():
        if isinstance(content, list):
            for line in content:
                print(f"â€¢ {line}")
        else:
            print(f"{section}: {content}")
