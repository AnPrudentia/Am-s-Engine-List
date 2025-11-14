from typing import List, Dict, Optional
from datetime import datetime
import uuid


class DriftTracker:
    def __init__(self):
        self.drift_log: List[Dict] = []

    def log_drift_event(self, origin_state: List[str], current_state: List[str], context: str = "") -> Dict:
        """Log a new drift event by comparing the origin and current state of identity or emotion."""
        lost_traits = [trait for trait in origin_state if trait not in current_state]
        gained_traits = [trait for trait in current_state if trait not in origin_state]

        # Safe severity calculation
        severity_denominator = len(origin_state) if origin_state else 1
        severity = len(lost_traits) / severity_denominator

        drift_event = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "context": context,
            "origin_state": origin_state.copy(),  # Copy to prevent external modification
            "current_state": current_state.copy(),
            "lost_traits": lost_traits,
            "gained_traits": gained_traits,
            "severity": round(severity, 4)  # Round for cleaner display
        }

        self.drift_log.append(drift_event)
        return drift_event

    def get_drift_history(self) -> List[Dict]:
        """Return the full history of logged drift events."""
        return self.drift_log.copy()  # Return copy to prevent external modification

    def get_recent_drift(self) -> Dict:
        """Return the most recent drift event, if any."""
        return self.drift_log[-1].copy() if self.drift_log else {}

    def get_drift_summary(self) -> Dict:
        """Get a summary of all drift events."""
        if not self.drift_log:
            return {}
        
        total_events = len(self.drift_log)
        total_lost = sum(len(event["lost_traits"]) for event in self.drift_log)
        total_gained = sum(len(event["gained_traits"]) for event in self.drift_log)
        avg_severity = sum(event["severity"] for event in self.drift_log) / total_events
        
        return {
            "total_drift_events": total_events,
            "total_lost_traits": total_lost,
            "total_gained_traits": total_gained,
            "average_severity": round(avg_severity, 4),
            "first_event": self.drift_log[0]["timestamp"],
            "last_event": self.drift_log[-1]["timestamp"]
        }


# Enhanced Demo
if __name__ == "__main__":
    tracker = DriftTracker()

    # Test case 1: Normal scenario
    baseline = ["creative", "resilient", "empathetic"]
    altered = ["resilient", "strategic", "guarded"]
    drift1 = tracker.log_drift_event(baseline, altered, context="Post-conflict self-assessment")

    # Test case 2: Empty origin state (should not cause division by zero)
    empty_baseline = []
    current = ["analytical", "cautious"]
    drift2 = tracker.log_drift_event(empty_baseline, current, context="Initial state definition")

    # Test case 3: Identical states (no drift)
    identical = ["happy", "confident"]
    drift3 = tracker.log_drift_event(identical, identical.copy(), context="Stable period")

    print("🧪 DRIFT TRACKER DEMO")
    print("=" * 50)
    
    # Show all events
    history = tracker.get_drift_history()
    for i, event in enumerate(history, 1):
        print(f"\nEvent #{i}:")
        print(f"  Context: {event['context']}")
        print(f"  Lost: {event['lost_traits'] or 'None'}")
        print(f"  Gained: {event['gained_traits'] or 'None'}")
        print(f"  Severity: {event['severity']:.2f}")

    # Show summary
    summary = tracker.get_drift_summary()
    print(f"\n📊 SUMMARY")
    print(f"Total events: {summary['total_drift_events']}")
    print(f"Total lost traits: {summary['total_lost_traits']}")
    print(f"Total gained traits: {summary['total_gained_traits']}")
    print(f"Average severity: {summary['average_severity']:.2f}")

    # Show most recent
    recent = tracker.get_recent_drift()
    print(f"\n🕒 MOST RECENT")
    print(f"Context: {recent['context']}")
    print(f"Severity: {recent['severity']:.2f}")