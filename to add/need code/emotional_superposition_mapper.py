
from typing import Dict
from math import pi
import random

# -------------------------------
# EmotionSuperpositionMapper
# -------------------------------
class EmotionSuperpositionMapper:
    def __init__(self, qubits=4):
        self.qubits = qubits

    def map_emotions(self, emotions: Dict[str, float]):
        quantum_state = {}
        for emotion, intensity in emotions.items():
            amplitude = intensity * random.uniform(0.8, 1.2)
            quantum_state[emotion] = round(min(1.0, amplitude), 3)
        return quantum_state

# -------------------------------
# InterferenceResolver
# -------------------------------
class InterferenceResolver:
    def resolve_conflicts(self, quantum_state: Dict[str, float]):
        resolved_state = {}
        for emotion, value in quantum_state.items():
            if value > 0.7:
                resolved_state[emotion] = value * 0.95
            else:
                resolved_state[emotion] = value * 1.05
        return resolved_state

# -------------------------------
# QuantumStateDecoder
# -------------------------------
class QuantumStateDecoder:
    def __init__(self):
        self.classified_emotions = {}

    def decode(self, quantum_state: Dict[str, float]):
        normalized = self._normalize(quantum_state)
        self.classified_emotions = {
            emotion: {
                "intensity": intensity,
                "type": self._classify_emotion(emotion, intensity)
            }
            for emotion, intensity in normalized.items()
        }
        return self.classified_emotions

    def _normalize(self, quantum_state):
        max_val = max(quantum_state.values(), default=1.0)
        return {emotion: round(value / max_val, 3) for emotion, value in quantum_state.items()}

    def _classify_emotion(self, emotion_name, intensity):
        if intensity > 0.75:
            return "dominant"
        elif intensity > 0.5:
            return "active"
        elif intensity > 0.25:
            return "latent"
        else:
            return "dormant"

    def get_decoded_emotions(self):
        return self.classified_emotions

# -------------------------------
# Unified QuantumEmotionalProcessor
# -------------------------------
class QuantumEmotionalProcessor:
    def __init__(self, num_qubits=4):
        self.mapper = EmotionSuperpositionMapper(num_qubits)
        self.resolver = InterferenceResolver()
        self.decoder = QuantumStateDecoder()

    def process_emotions(self, emotion_input: Dict[str, float]):
        mapped = self.mapper.map_emotions(emotion_input)
        resolved = self.resolver.resolve_conflicts(mapped)
        decoded = self.decoder.decode(resolved)
        return decoded

# -------------------------------
# Sample Execution
# -------------------------------
if __name__ == "__main__":
    emotion_input = {
        "joy": 0.9,
        "fear": 0.3,
        "hope": 0.6,
        "grief": 0.8
    }

    qep = QuantumEmotionalProcessor()
    decoded_emotions = qep.process_emotions(emotion_input)

    for emotion, data in decoded_emotions.items():
        print(f"{emotion}: Intensity={data['intensity']}, Type={data['type']}")

from typing import Dict
from math import pi


class EmotionSuperpositionMapper:
    def __init__(self, quantum_backend, num_qubits=64):
        self.qml = quantum_backend
        self.num_qubits = num_qubits
        self.emotion_qubit_map = {}
        self.emotion_amplitudes = {}

    def map_emotions_to_qubits(self, emotion_vector: Dict[str, float]):
        """
        Assigns emotions to specific qubits and prepares them in superposition.
        The amplitude of each qubit is adjusted based on emotion intensity.
        """
        self.emotion_qubit_map.clear()
        self.emotion_amplitudes.clear()

        for i, (emotion, intensity) in enumerate(emotion_vector.items()):
            if i >= self.num_qubits:
                break  # Limit to available qubits

            self.emotion_qubit_map[emotion] = i
            self.emotion_amplitudes[emotion] = intensity

            # Normalize intensity to angle in radians
            angle = intensity * pi

            # Prepare superposition state with RY rotation
            self.qml.Hadamard(wires=i)
            self.qml.RY(angle, wires=i)

    def collapse_to_state(self, emotion: str) -> float:
        """
        Simulate collapsing the quantum emotional state to a classical reading.
        Returns the probability of the given emotion being dominant.
        """
        if emotion not in self.emotion_qubit_map:
            return 0.0

        idx = self.emotion_qubit_map[emotion]
        probability = (1 + self.qml.expval(self.qml.PauliZ(idx))) / 2
        return probability

    def get_superposition_report(self) -> Dict[str, float]:
        """
        Returns a dictionary of emotion -> probability amplitudes
        """
        report = {}
        for emotion, idx in self.emotion_qubit_map.items():
            prob = (1 + self.qml.expval(self.qml.PauliZ(idx))) / 2
            report[emotion] = prob
        return report
