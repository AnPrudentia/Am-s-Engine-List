
from typing import List, Dict
from datetime import datetime
import uuid


class BehaviorProjectionUnit:
    def __init__(self):
        self.behavior_records: List[Dict] = []

    def project_future_behavior(self, recent_traits: List[str], context: str) -> Dict:
        """
        Project future behavior based on recent identity traits and context.
        Uses pattern recognition and symbolic inference.
        """
        projected_outcome = self._infer_behavior_pattern(recent_traits, context)

        projection = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "input_traits": recent_traits,
            "context": context,
            "projection": projected_outcome
        }

        self.behavior_records.append(projection)
        return projection

    def _infer_behavior_pattern(self, traits: List[str], context: str) -> str:
        if "isolated" in traits or "withdrawn" in traits:
            if "pressure" in context:
                return "Likely to retreat or suppress feelings further under stress"
            return "May default to avoidance or internalized response"

        if "agitated" in traits or "impulsive" in traits:
            return "Behavior may escalate quickly in reactive contexts"

        if "resilient" in traits or "reflective" in traits:
            return "May pause, regroup, and adapt strategy with context shifts"

        if "empathetic" in traits:
            return "May prioritize others' needs even at emotional cost"

        return "Behavior uncertainâ€”traits suggest multiple pathways"

    def get_projection_history(self) -> List[Dict]:
        return self.behavior_records


# Demo
if __name__ == "__main__":
    projector = BehaviorProjectionUnit()
    traits = ["resilient", "empathetic", "isolated"]
    context = "emotional conflict under pressure"

    result = projector.project_future_behavior(traits, context)

    print("ðŸ§­ BEHAVIOR PROJECTION RESULT")
    print("Projected Behavior:", result["projection"])
    print("Context:", result["context"])

