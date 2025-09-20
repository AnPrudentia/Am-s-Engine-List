from typing import List, Dict
from datetime import datetime
import uuid


class ShatteredSelfEchoDetector:
    def __init__(self):
        self.echo_log: List[Dict] = []

    def detect_echo(self, thought_fragments: List[str]) -> Dict:
        """
        Detect echo patterns that indicate signs of a fractured or fragmented self.
        These patterns include repetition of self-negating phrases, conflicting identities,
        or dissociative expressions.
        """
        echo_id = f"ECHO-{uuid.uuid4().hex[:8]}"
        matched_fragments = []
        contradiction_pairs = []

        for i, fragment in enumerate(thought_fragments):
            for j in range(i + 1, len(thought_fragments)):
                other = thought_fragments[j]
                if self._is_self_contradictory(fragment, other):
                    contradiction_pairs.append((fragment, other))
                    matched_fragments.extend([fragment, other])

        matched_fragments = list(set(matched_fragments))

        echo_report = {
            "id": echo_id,
            "timestamp": datetime.utcnow().isoformat(),
            "total_fragments": len(thought_fragments),
            "contradictions_found": len(contradiction_pairs),
            "matched_fragments": matched_fragments,
            "contradiction_pairs": contradiction_pairs
        }

        self.echo_log.append(echo_report)
        return echo_report

    def _is_self_contradictory(self, a: str, b: str) -> bool:
        """
        Naive contradiction detection based on common linguistic negations.
        Future upgrade: semantic contradiction engine.
        """
        return (("not" in a.lower() and b.lower().replace("not ", "") in a.lower())
                or ("never" in a.lower() and b.lower().replace("never ", "") in a.lower())
                or (a.lower() in b.lower() and "but" in b.lower()))

    def history(self) -> List[Dict]:
        """Return the log of echo detections."""
        return self.echo_log


# Demo
if __name__ == "__main__":
    detector = ShatteredSelfEchoDetector()
    fragments = [
        "I want to be seen",
        "I never want anyone to look at me",
        "I am strong",
        "But I am not enough"
    ]
    echo = detector.detect_echo(fragments)
    print("ðŸ§  Echo Report:")
    for key, val in echo.items():
        print(f"{key}: {val}")
