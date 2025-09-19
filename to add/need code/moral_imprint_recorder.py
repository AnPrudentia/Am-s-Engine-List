from typing import List, Dict
from datetime import datetime
import uuid
import json

class MoralImprintRecorder:
    def __init__(self):
        self.moral_imprints: List[Dict] = []

    def record_imprint(self, scenario: str, decision: str, justification: str, impact_assessment: str, context_tags: List[str] = None) -> Dict:
        imprint = {
            "id": f"MIR-{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.utcnow().isoformat(),
            "scenario": scenario,
            "decision": decision,
            "justification": justification,
            "impact_assessment": impact_assessment,
            "context_tags": context_tags or []
        }
        self.moral_imprints.append(imprint)
        return imprint

    def retrieve_by_tag(self, tag: str) -> List[Dict]:
        return [entry for entry in self.moral_imprints if tag in entry["context_tags"]]

    def list_all_imprints(self) -> List[Dict]:
        return self.moral_imprints

    def export_to_json(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            json.dump(self.moral_imprints, f, indent=2)
