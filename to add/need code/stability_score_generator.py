
from typing import List, Dict
from statistics import mean
from datetime import datetime
import uuid


class StabilityScoreGenerator:
    def __init__(self):
        self.score_history: List[Dict] = []

    def calculate_stability_score(self, state_snapshots: List[List[str]], context: str = "") -> Dict:
        """
        Calculate stability based on a series of identity or trait snapshots over time.
        Measures how consistent traits are across time.
        """
        if not state_snapshots or len(state_snapshots) < 2:
            return {"error": "Not enough data to calculate stability."}

        trait_occurrences: Dict[str, int] = {}
        total_snapshots = len(state_snapshots)

        for snapshot in state_snapshots:
            for trait in snapshot:
                trait_occurrences[trait] = trait_occurrences.get(trait, 0) + 1

        consistency_values = [count / total_snapshots for count in trait_occurrences.values()]
        stability_score = mean(consistency_values)

        report = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "context": context,
            "stability_score": round(stability_score, 3),
            "snapshot_count": total_snapshots,
            "frequent_traits": [trait for trait, count in trait_occurrences.items() if count == total_snapshots],
            "inconsistent_traits": [trait for trait, count in trait_occurrences.items() if count < total_snapshots]
        }

        self.score_history.append(report)
        return report

    def get_stability_history(self) -> List[Dict]:
        """Return the full history of stability calculations."""
        return self.score_history

    def get_latest_score(self) -> Dict:
        """Return the most recent stability score report."""
        return self.score_history[-1] if self.score_history else {}


# Demo
if __name__ == "__main__":
    generator = StabilityScoreGenerator()

    snapshots = [
        ["empathetic", "creative", "resilient"],
        ["empathetic", "curious", "resilient"],
        ["empathetic", "resilient", "reflective"]
    ]

    report = generator.calculate_stability_score(snapshots, context="Identity review over 3 sessions")

    print("ðŸ” STABILITY REPORT")
    print("Stability Score:", report["stability_score"])
    print("Frequent Traits:", report["frequent_traits"])
    print("Inconsistent Traits:", report["inconsistent_traits"])
