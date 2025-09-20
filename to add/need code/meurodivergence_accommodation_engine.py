from typing import Dict, List
from datetime import datetime
import uuid


class NeurodivergenceAccommodationEngine:
    def __init__(self):
        self.accommodations: List[Dict] = []

    def generate_accommodation(self, profile_traits: Dict[str, bool], context: str = "") -> Dict:
        accommodation_id = f"NDAE-{uuid.uuid4().hex[:8]}"
        recommendations = self._build_recommendations(profile_traits)

        result = {
            "id": accommodation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "profile_traits": profile_traits,
            "context": context,
            "recommendations": recommendations
        }

        self.accommodations.append(result)
        return result

    def _build_recommendations(self, traits: Dict[str, bool]) -> List[str]:
        recs = []
        if traits.get("adhd"):
            recs.append("Break tasks into short, focused intervals (e.g., Pomodoro method).")
            recs.append("Use visual reminders and checklists.")
        if traits.get("autism"):
            recs.append("Provide clear, predictable routines.")
            recs.append("Minimize sensory overload (light, sound, touch).")
        if traits.get("anxiety"):
            recs.append("Allow extra processing time and create a calm environment.")
            recs.append("Use reassuring, supportive language.")
        if traits.get("dyslexia"):
            recs.append("Offer text-to-speech options and visual aids.")
            recs.append("Avoid heavy reliance on written instructions alone.")
        if not recs:
            recs.append("No traits detected; proceed with general accessibility support.")
        return recs

    def list_accommodations(self) -> List[Dict]:
        return self.accommodations
