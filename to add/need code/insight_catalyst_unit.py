from typing import List, Dict
from datetime import datetime
import uuid


class InsightCatalystUnit:
    def __init__(self):
        self.insight_log: List[Dict] = []

    def catalyze(self, input_data: List[str], contradiction_context: List[str] = None) -> Dict:
        """
        Generate insight by drawing connections between multiple data points,
        especially in the presence of contradiction or ambiguity.
        """
        insight_id = f"INSIGHT-{uuid.uuid4().hex[:8]}"
        merged_insight = " | ".join(sorted(set(input_data)))

        if contradiction_context:
            merged_insight += " || ⚡ CONTRADICTION CONTEXT: " + ", ".join(contradiction_context)

        insight = {
            "id": insight_id,
            "timestamp": datetime.utcnow().isoformat(),
            "source_data": input_data,
            "contradiction_context": contradiction_context,
            "generated_insight": merged_insight
        }

        self.insight_log.append(insight)
        return insight

    def list_insights(self) -> List[Dict]:
        """Return all generated insights."""
        return self.insight_log


# Demo
if __name__ == "__main__":
    icu = InsightCatalystUnit()
    data_points = [
        "He avoids conflict",
        "He steps into every argument",
        "He wants peace"
    ]
    contradictions = ["He avoids conflict", "He steps into every argument"]

    insight = icu.catalyze(data_points, contradictions)
    print("✨ Insight Generated:")
    print(insight["generated_insight"])
