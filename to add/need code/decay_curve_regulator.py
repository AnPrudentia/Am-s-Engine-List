
import math
from datetime import datetime, timedelta
import uuid

class DecayCurveRegulator:
    def __init__(self, half_life_hours=48.0):
        self.half_life = half_life_hours
        self.decay_base = 0.5 ** (1.0 / self.half_life)
        self.entries = []

    def register_event(self, content: str, intensity: float) -> str:
        event_id = str(uuid.uuid4())
        self.entries.append({
            "id": event_id,
            "content": content,
            "intensity": intensity,
            "timestamp": datetime.utcnow()
        })
        return event_id

    def _calculate_decay(self, initial_intensity, hours_elapsed):
        return initial_intensity * (self.decay_base ** hours_elapsed)

    def get_adjusted_intensity(self, event_id: str) -> float:
        for entry in self.entries:
            if entry["id"] == event_id:
                elapsed = (datetime.utcnow() - entry["timestamp"]).total_seconds() / 3600
                return round(self._calculate_decay(entry["intensity"], elapsed), 4)
        raise ValueError("Event not found")

    def decay_report(self):
        report = []
        now = datetime.utcnow()
        for entry in self.entries:
            hours_elapsed = (now - entry["timestamp"]).total_seconds() / 3600
            adjusted = self._calculate_decay(entry["intensity"], hours_elapsed)
            report.append({
                "id": entry["id"],
                "content": entry["content"],
                "original_intensity": entry["intensity"],
                "adjusted_intensity": round(adjusted, 4),
                "hours_elapsed": round(hours_elapsed, 2)
            })
        return report

# Demo
if __name__ == "__main__":
    regulator = DecayCurveRegulator(half_life_hours=24)

    eid1 = regulator.register_event("Trauma from past failure", 0.9)
    eid2 = regulator.register_event("Joyful memory from vacation", 0.7)

    print("\nðŸ“‰ Initial Decay Report:")
    report = regulator.decay_report()
    for r in report:
        print(f"- {r['content']}: {r['adjusted_intensity']} (after {r['hours_elapsed']} hours)")

    # Simulate time passage manually if desired
