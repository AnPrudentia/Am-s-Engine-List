
import numpy as np
import uuid
from datetime import datetime
from typing import Dict, List

class QuantumMoralOptimizer:
    def __init__(self):
        self.ethical_frameworks = {
            "utilitarianism": self._utilitarian_score,
            "deontology": self._deontological_score,
            "virtue_ethics": self._virtue_score
        }
        self.moral_decision_log: List[Dict] = []

    def evaluate_action(self, action_description: str, context: Dict, ethical_weights: Dict) -> Dict:
        """
        Evaluate the moral weight of an action using multiple ethical models.
        """
        total_score = 0.0
        weight_sum = 0.0
        model_scores = {}

        for model, weight in ethical_weights.items():
            if model in self.ethical_frameworks:
                score = self.ethical_frameworks[model](action_description, context)
                model_scores[model] = score
                total_score += score * weight
                weight_sum += weight

        composite_score = total_score / weight_sum if weight_sum > 0 else 0

        decision = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "action": action_description,
            "context": context,
            "model_scores": model_scores,
            "composite_score": round(composite_score, 3)
        }

        self.moral_decision_log.append(decision)
        return decision

    def _utilitarian_score(self, action_description: str, context: Dict) -> float:
        """
        Estimate net benefit to well-being on a 0â€“1 scale.
        """
        benefit = context.get("expected_benefit", 0.5)
        harm = context.get("expected_harm", 0.5)
        return max(0.0, min(1.0, benefit - harm + 0.5))

    def _deontological_score(self, action_description: str, context: Dict) -> float:
        """
        Binary rule-following: 1 if aligns with duty, 0 otherwise.
        """
        duties = context.get("duties", [])
        action_rules = context.get("rules", [])
        return 1.0 if set(duties).issubset(set(action_rules)) else 0.0

    def _virtue_score(self, action_description: str, context: Dict) -> float:
        """
        Degree to which the action reflects core virtues (e.g., honesty, courage).
        """
        virtues = context.get("virtues", [])
        expressed = context.get("expressed_virtues", [])
        match_count = len(set(virtues) & set(expressed))
        return match_count / len(virtues) if virtues else 0.0

    def get_moral_history(self) -> List[Dict]:
        return self.moral_decision_log

# Demo
if __name__ == "__main__":
    optimizer = QuantumMoralOptimizer()
    context = {
        "expected_benefit": 0.7,
        "expected_harm": 0.2,
        "duties": ["protect"],
        "rules": ["protect", "inform"],
        "virtues": ["honesty", "courage"],
        "expressed_virtues": ["honesty", "loyalty"]
    }
    weights = {
        "utilitarianism": 0.5,
        "deontology": 0.3,
        "virtue_ethics": 0.2
    }
    result = optimizer.evaluate_action("Divulge information to protect others", context, weights)
    print("âš–ï¸ Moral Decision Evaluation:")
    print(result)