
from typing import Dict, List
from datetime import datetime
import uuid


class CompassionLoadDistributionEngine:
    def __init__(self):
        self.distribution_log: List[Dict] = []

    def distribute_load(self, emotional_burden: float, available_units: List[str]) -> Dict:
        """
        Distribute emotional burden evenly across available emotional support units.
        Each unit receives a portion of the load based on total availability.
        """
        if not available_units:
            raise ValueError("No units available for load distribution.")

        distribution_id = f"CLOAD-{uuid.uuid4().hex[:8]}"
        timestamp = datetime.utcnow().isoformat()

        portion = round(emotional_burden / len(available_units), 4)
        distribution_map = {unit: portion for unit in available_units}

        record = {
            "id": distribution_id,
            "timestamp": timestamp,
            "total_burden": emotional_burden,
            "units": available_units,
            "distribution": distribution_map
        }

        self.distribution_log.append(record)
        return record

    def list_distributions(self) -> List[Dict]:
        """Return the full history of emotional load distributions."""
        return self.distribution_log


# Demo
if __name__ == "__main__":
    engine = CompassionLoadDistributionEngine()
    burden = 4.5
    units = ["Anima", "Umbra", "Luma"]

    result = engine.distribute_load(burden, units)
    print("ðŸ’ž Emotional Load Distribution Complete:")
    for unit, portion in result["distribution"].items():
        print(f"{unit}: {portion}")
