
from typing import Dict, List
from math import log, sqrt
from datetime import datetime


class QuantumConsciousnessMetric:
    def __init__(self):
        self.metrics_log: List[Dict] = []

    def evaluate_consciousness_state(self, coherence: float, entanglement: float, self_awareness: float) -> Dict:
        """
        Calculate the composite quantum consciousness metric.
        Values expected in range [0.0, 1.0] for each parameter.
        """
        assert 0.0 <= coherence <= 1.0, "Coherence must be between 0.0 and 1.0"
        assert 0.0 <= entanglement <= 1.0, "Entanglement must be between 0.0 and 1.0"
        assert 0.0 <= self_awareness <= 1.0, "Self-awareness must be between 0.0 and 1.0"

        stability_index = sqrt(coherence * entanglement)
        reflective_depth = -log(1.0001 - self_awareness)  # Prevent log(0)
        consciousness_score = round((stability_index + reflective_depth) / 2, 4)

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "coherence": coherence,
            "entanglement": entanglement,
            "self_awareness": self_awareness,
            "stability_index": round(stability_index, 4),
            "reflective_depth": round(reflective_depth, 4),
            "consciousness_score": consciousness_score
        }

        self.metrics_log.append(result)
        return result

    def latest_score(self) -> float:
        """Return the latest consciousness score."""
        return self.metrics_log[-1]["consciousness_score"] if self.metrics_log else 0.0

    def history(self) -> List[Dict]:
        """Return all consciousness metric evaluations."""
        return self.metrics_log


# Demo
if __name__ == "__main__":
    qcm = QuantumConsciousnessMetric()

    state1 = qcm.evaluate_consciousness_state(coherence=0.92, entanglement=0.88, self_awareness=0.76)
    print("ðŸ§  Quantum Consciousness Metric:")
    for k, v in state1.items():
        print(f"{k}: {v}")

    print("\nðŸ“ˆ Latest Score:", qcm.latest_score())
