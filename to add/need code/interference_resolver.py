from typing import Dict, List, Tuple
from datetime import datetime
from math import pi


class InterferenceResolver:
    def __init__(self, quantum_backend, threshold: float = 0.3):
        self.qml = quantum_backend
        self.threshold = threshold
        self.conflict_log = []

    def detect_interference(self, emotion_vector: Dict[str, float], conflicting_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Detects conflicting emotional states based on predefined opposing emotion pairs.
        Returns a list of conflicting pairs with significant overlap.
        """
        conflicts = []
        for emotion_a, emotion_b in conflicting_pairs:
            val_a = emotion_vector.get(emotion_a, 0.0)
            val_b = emotion_vector.get(emotion_b, 0.0)

            if abs(val_a - val_b) <= self.threshold and (val_a + val_b) > 1.0:
                conflicts.append((emotion_a, emotion_b))
                self.conflict_log.append({
                    "pair": (emotion_a, emotion_b),
                    "values": (val_a, val_b),
                    "timestamp": datetime.utcnow()
                })
        return conflicts

    def resolve_conflict(self, emotion_a: str, emotion_b: str, emotion_vector: Dict[str, float]) -> Dict[str, float]:
        """
        Applies quantum-based interference resolution to overlapping emotions.
        Adjusts amplitudes via quantum rotations to favor contextual emotion.
        """
        idx_a = hash(emotion_a) % 64
        idx_b = hash(emotion_b) % 64

        val_a = emotion_vector.get(emotion_a, 0.0)
        val_b = emotion_vector.get(emotion_b, 0.0)

        # Determine dominant
        dominant = emotion_a if val_a >= val_b else emotion_b
        suppression = emotion_b if dominant == emotion_a else emotion_a

        # Apply interference resolution gate
        self.qml.CRX(abs(val_a - val_b) * pi, wires=[idx_a, idx_b])
        self.qml.RZ(pi / 2, wires=idx_b)

        # Adjust values directly for deterministic model
        resolved_vector = emotion_vector.copy()
        resolved_vector[dominant] = min(1.0, (val_a + val_b) / 1.5)
        resolved_vector[suppression] = max(0.0, (val_a + val_b) / 3.0)

        return resolved_vector

    def log_conflicts(self) -> List[Dict[str, any]]:
        """
        Returns all logged emotional conflicts for historical tracking or debugging.
        """
        return self.conflict_log
