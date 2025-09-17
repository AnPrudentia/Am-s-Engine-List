from typing import Dict, List
from datetime import datetime
import uuid


class SelfReconciliationEngine:
    def __init__(self):
        self.reconciliation_log: List[Dict] = []

    def reconcile(self, conflicting_selfs: List[str], context: str = "") -> Dict:
        resolution_id = f"RECON-{uuid.uuid4().hex[:8]}"
        harmonized_self = " âŸ¶ ".join(conflicting_selfs)
        summary = f"Harmonized trajectory from {conflicting_selfs[0]} to {conflicting_selfs[-1]}"             if len(conflicting_selfs) > 1 else "Single-state reflection"

        result = {
            "id": resolution_id,
            "timestamp": datetime.utcnow().isoformat(),
            "conflicting_selfs": conflicting_selfs,
            "harmonized_path": harmonized_self,
            "summary": summary,
            "context": context
        }

        self.reconciliation_log.append(result)
        return result

    def list_resolutions(self) -> List[Dict]:
        return self.reconciliation_log
