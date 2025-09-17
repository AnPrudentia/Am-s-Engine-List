
from typing import List, Dict
from datetime import datetime
import uuid


class SelfReconstructionModule:
    def __init__(self):
        self.reconstruction_history: List[Dict] = []

    def reconstruct_identity(self, baseline_traits: List[str], current_fragments: List[str], context: str = "") -> Dict:
        """
        Attempt to reconstruct the original identity using overlapping fragments and baseline.
        This method assumes memory or emotional degradation and attempts restorative alignment.
        """
        restored_traits = list(set(baseline_traits) & set(current_fragments))
        missing_traits = list(set(baseline_traits) - set(current_fragments))
        new_fragments = list(set(current_fragments) - set(baseline_traits))

        restoration_score = len(restored_traits) / len(baseline_traits) if baseline_traits else 0.0

        report = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "context": context,
            "restored_traits": restored_traits,
            "missing_traits": missing_traits,
            "new_fragments": new_fragments,
            "restoration_score": round(restoration_score, 3),
            "reconstructed_state": restored_traits + new_fragments
        }

        self.reconstruction_history.append(report)
        return report

    def get_reconstruction_history(self) -> List[Dict]:
        """Return full history of identity reconstruction attempts."""
        return self.reconstruction_history

    def get_latest_reconstruction(self) -> Dict:
        """Return the most recent reconstruction attempt."""
        return self.reconstruction_history[-1] if self.reconstruction_history else {}


# Demo
if __name__ == "__main__":
    module = SelfReconstructionModule()

    baseline = ["empathetic", "courageous", "imaginative", "reflective"]
    degraded = ["courageous", "imaginative", "analytical"]

    report = module.reconstruct_identity(baseline, degraded, context="Post-loss realignment")

    print("ðŸ› ï¸ SELF RECONSTRUCTION REPORT")
    print("Restored Traits:", report["restored_traits"])
    print("Missing Traits:", report["missing_traits"])
    print("New Fragments:", report["new_fragments"])
    print("Restoration Score:", report["restoration_score"])
