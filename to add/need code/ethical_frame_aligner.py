from typing import Dict, List
from datetime import datetime
import uuid


class EthicalFrameAligner:
    def __init__(self):
        self.alignments: List[Dict] = []

    def align_ethics(self, input_behavior: str, source_ethics: List[str], target_ethics: List[str]) -> Dict:
        alignment_id = f"EFA-{uuid.uuid4().hex[:8]}"
        conflicts, harmonies = self._compare_frameworks(source_ethics, target_ethics)

        alignment = {
            "id": alignment_id,
            "timestamp": datetime.utcnow().isoformat(),
            "behavior": input_behavior,
            "source_ethics": source_ethics,
            "target_ethics": target_ethics,
            "conflicts": conflicts,
            "harmonies": harmonies,
            "alignment_note": f"Behavior adapted for target ethical context"
        }

        self.alignments.append(alignment)
        return alignment

    def _compare_frameworks(self, source: List[str], target: List[str]) -> (List[str], List[str]):
        harmonies = list(set(source).intersection(set(target)))
        conflicts = list(set(source).symmetric_difference(set(target)))
        return conflicts, harmonies

    def list_alignments(self) -> List[Dict]:
        return self.alignments
