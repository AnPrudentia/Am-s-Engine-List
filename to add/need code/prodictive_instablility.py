from typing import List, Dict
from datetime import datetime
import uuid
import random
import matplotlib.pyplot as plt
import pandas as pd

class PredictiveInstabilityMonitor:
    def __init__(self):
        self.history: List[Dict] = []

    def assess_instability(self, emotion_sequence: List[Dict[str, float]]) -> Dict:
        """
        Assess emotional instability across a time-sequenced list of emotional states.
        Each emotional state is a dictionary of emotions and intensities.
        """
        instability_score = 0.0
        trajectory = []

        for i in range(1, len(emotion_sequence)):
            prev = emotion_sequence[i - 1]
            curr = emotion_sequence[i]
            delta = sum(abs(curr.get(emotion, 0.0) - prev.get(emotion, 0.0)) for emotion in set(prev) | set(curr))
            instability_score += delta
            trajectory.append(delta)

        avg_instability = instability_score / max(1, len(trajectory))
        entry = {
            "id": f"PIM-{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.utcnow().isoformat(),
            "sequence_length": len(emotion_sequence),
            "instability_score": round(instability_score, 4),
            "average_shift": round(avg_instability, 4),
            "trajectory": trajectory
        }
        self.history.append(entry)
        return entry

    def plot_trajectory(self, trajectory: List[float]):
        """
        Plot the instability trajectory over time.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(trajectory, marker='o', linestyle='-', alpha=0.8)
        plt.title("Emotional Instability Trajectory")
        plt.xlabel("Time Step")
        plt.ylabel("Shift Magnitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
