from typing import Dict, List
from datetime import datetime
import uuid

class ContradictionWeightBalancer:
    def __init__(self):
        self.records: List[Dict] = []

    def evaluate_contradiction(self, statement_a: str, statement_b: str, context: str = "") -> Dict:
        weight_a = self._assess_weight(statement_a)
        weight_b = self._assess_weight(statement_b)

        dominant = statement_a if weight_a > weight_b else statement_b

        record = {
            "id": f"CWB-{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.utcnow().isoformat(),
            "statement_a": statement_a,
            "statement_b": statement_b,
            "context": context,
            "weight_a": weight_a,
            "weight_b": weight_b,
            "dominant": dominant
        }

        self.records.append(record)
        return record

    def _assess_weight(self, statement: str) -> float:
        base = len(statement) / 100
        sentiment_weight = 0.5 if "not" in statement or "never" in statement else 1.0
        return round(min(1.0, base * sentiment_weight), 3)

    def list_contradictions(self) -> List[Dict]:
        return self.records
