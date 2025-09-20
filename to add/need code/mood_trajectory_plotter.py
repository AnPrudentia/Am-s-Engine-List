
from typing import List, Dict
from datetime import datetime
import matplotlib.pyplot as plt


class MoodTrajectoryPlotter:
    def __init__(self):
        self.trajectory_log: List[Dict] = []

    def log_mood(self, timestamp: datetime, mood_scores: Dict[str, float]) -> None:
        """
        Log mood scores at a given timestamp.
        """
        self.trajectory_log.append({
            "timestamp": timestamp,
            "mood_scores": mood_scores
        })

    def plot_trajectory(self) -> None:
        """
        Plot the mood trajectory over time for each tracked mood dimension.
        """
        if not self.trajectory_log:
            print("âš ï¸ No mood data to plot.")
            return

        timestamps = [entry["timestamp"] for entry in self.trajectory_log]
        mood_dimensions = self.trajectory_log[0]["mood_scores"].keys()
        mood_data = {mood: [] for mood in mood_dimensions}

        for entry in self.trajectory_log:
            for mood in mood_dimensions:
                mood_data[mood].append(entry["mood_scores"].get(mood, 0))

        plt.figure(figsize=(12, 6))
        for mood, values in mood_data.items():
            plt.plot(timestamps, values, label=mood)

        plt.xlabel("Time")
        plt.ylabel("Mood Intensity")
        plt.title("Mood Trajectory Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Demo
if __name__ == "__main__":
    import time

    mtp = MoodTrajectoryPlotter()
    now = datetime.utcnow()

    # Simulated mood data
    mtp.log_mood(now, {"joy": 0.6, "anxiety": 0.4, "focus": 0.7})
    time.sleep(1)
    mtp.log_mood(datetime.utcnow(), {"joy": 0.7, "anxiety": 0.3, "focus": 0.8})
    time.sleep(1)
    mtp.log_mood(datetime.utcnow(), {"joy": 0.65, "anxiety": 0.35, "focus": 0.75})

    mtp.plot_trajectory()
