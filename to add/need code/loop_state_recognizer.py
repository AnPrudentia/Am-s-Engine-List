from typing import List, Dict
from datetime import datetime
import uuid

class LoopStateRecognizer:
    def __init__(self):
        self.detected_loops: List[Dict] = []

    def analyze_behavior_sequence(self, sequence: List[str], tolerance: int = 2) -> Dict:
        loop_patterns = {}
        loop_id = f"LOOP-{uuid.uuid4().hex[:8]}"
        detected = False

        for i in range(len(sequence)):
            for j in range(i + 1, len(sequence)):
                window = sequence[i:j]
                if len(window) < 2:
                    continue
                repetitions = self._count_repetitions(sequence, window)
                if repetitions >= tolerance:
                    pattern = tuple(window)
                    loop_patterns[pattern] = repetitions
                    detected = True

        loop_record = {
            "id": loop_id,
            "timestamp": datetime.utcnow().isoformat(),
            "sequence_length": len(sequence),
            "patterns": {str(k): v for k, v in loop_patterns.items()},
            "loop_detected": detected
        }

        self.detected_loops.append(loop_record)
        return loop_record

    def _count_repetitions(self, sequence: List[str], pattern: List[str]) -> int:
        pattern_len = len(pattern)
        count = 0
        for i in range(0, len(sequence) - pattern_len + 1):
            if sequence[i:i + pattern_len] == pattern:
                count += 1
        return count

    def get_all_loops(self) -> List[Dict]:
        return self.detected_loops
