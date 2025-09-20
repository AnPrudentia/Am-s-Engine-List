from typing import Dict, List
from datetime import datetime
import uuid


class LearningModalityDetector:
    def __init__(self):
        self.detections: List[Dict] = []

    def detect_modality(self, interactions: List[str]) -> Dict:
        detection_id = f"LMD-{uuid.uuid4().hex[:8]}"
        modality = self._determine_modality(interactions)

        detection = {
            "id": detection_id,
            "timestamp": datetime.utcnow().isoformat(),
            "interactions": interactions,
            "detected_modality": modality
        }

        self.detections.append(detection)
        return detection

    def _determine_modality(self, interactions: List[str]) -> str:
        modality_keywords = {
            "visual": ["look", "see", "picture", "diagram", "graph"],
            "auditory": ["talk", "listen", "hear", "discussion", "speak"],
            "tactile": ["touch", "feel", "hands-on", "manipulate", "build"],
            "reading/writing": ["read", "write", "type", "note", "journal"]
        }

        scores = {modality: 0 for modality in modality_keywords}
        for interaction in interactions:
            for modality, keywords in modality_keywords.items():
                if any(keyword in interaction.lower() for keyword in keywords):
                    scores[modality] += 1

        return max(scores, key=scores.get)

    def list_detections(self) -> List[Dict]:
        return self.detections
