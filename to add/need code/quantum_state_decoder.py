from typing import Dict, List
from math import pi


class QuantumStateDecoder:
    def __init__(self, quantum_backend, decoding_threshold: float = 0.6):
        self.qml = quantum_backend
        self.threshold = decoding_threshold

    def decode_emotional_state(self, qubit_emotion_map: Dict[int, str], snapshot: List[float]) -> Dict[str, float]:
        """
        Decodes a quantum state snapshot into a classical emotion vector.
        Each index in snapshot corresponds to a qubit's Z-basis measurement result (-1 to 1).
        """
        decoded_emotions = {}
        for idx, z_value in enumerate(snapshot):
            emotion = qubit_emotion_map.get(idx)
            if emotion:
                # Convert from Z-basis result to probability [0,1]
                prob = (1 + z_value) / 2
                decoded_emotions[emotion] = round(prob, 4)
        return decoded_emotions

    def filter_significant_emotions(self, emotion_vector: Dict[str, float]) -> Dict[str, float]:
        """
        Filters decoded emotions to return only those above the decoding threshold.
        """
        return {e: v for e, v in emotion_vector.items() if v >= self.threshold}

    def normalize_emotions(self, emotion_vector: Dict[str, float]) -> Dict[str, float]:
        """
        Normalizes emotion values to sum to 1 for probabilistic interpretations.
        """
        total = sum(emotion_vector.values())
        if total == 0:
            return {e: 0.0 for e in emotion_vector}
        return {e: round(v / total, 4) for e, v in emotion_vector.items()}

    def classify_emotional_profile(self, emotion_vector: Dict[str, float]) -> str:
        """
        Assigns a simplified emotional state classification based on dominant categories.
        Can be expanded into multi-axis interpretation (e.g. valence/arousal).
        """
        if not emotion_vector:
            return "neutral"

        dominant_emotion = max(emotion_vector.items(), key=lambda x: x[1])[0]

        # Simplified mapping
        positive = {"joy", "trust", "gratitude", "serenity", "love"}
        negative = {"anger", "fear", "sadness", "disgust", "shame"}

        if dominant_emotion in positive:
            return "positive-dominant"
        elif dominant_emotion in negative:
            return "negative-dominant"
        else:
            return "mixed or uncertain”
