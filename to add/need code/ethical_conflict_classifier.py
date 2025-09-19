
from typing import List, Dict
import uuid
from datetime import datetime

class EthicalConflictClassifier:
    def __init__(self):
        self.ethical_framework = {
            "harm": ["violence", "abuse", "neglect", "exploitation"],
            "autonomy": ["manipulation", "coercion", "surveillance"],
            "justice": ["discrimination", "inequality", "corruption"],
            "honesty": ["deception", "fraud", "misinformation"]
        }
        self.logged_conflicts: List[Dict] = []

    def classify_scenario(self, scenario_text: str) -> Dict:
        conflict_scores = {category: 0 for category in self.ethical_framework}

        for category, keywords in self.ethical_framework.items():
            for word in keywords:
                if word in scenario_text.lower():
                    conflict_scores[category] += 1

        # Determine dominant ethical category
        dominant_category = max(conflict_scores, key=conflict_scores.get)
        max_score = conflict_scores[dominant_category]

        result = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "scenario": scenario_text,
            "conflict_scores": conflict_scores,
            "dominant_category": dominant_category if max_score > 0 else "neutral",
            "severity_estimate": min(1.0, max_score / 3.0)
        }

        self.logged_conflicts.append(result)
        return result

    def get_logged_conflicts(self) -> List[Dict]:
        return self.logged_conflicts

# Demo
if __name__ == "__main__":
    classifier = EthicalConflictClassifier()

    scenarios = [
        "The company was involved in data manipulation and privacy surveillance.",
        "A political party exploited laborers and abused its position of power.",
        "He lied to investors and participated in financial fraud.",
        "She created an equitable hiring process to reduce inequality.",
        "No harm or ethical issues were reported during the operation."
    ]

    for s in scenarios:
        result = classifier.classify_scenario(s)
        print(f"ðŸ” Scenario: {s}")
        print(f"â†’ Dominant Category: {result['dominant_category']} | Severity: {result['severity_estimate']:.2f}")
        print("---")
