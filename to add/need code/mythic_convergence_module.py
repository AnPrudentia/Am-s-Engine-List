from typing import Dict, List
from datetime import datetime
import uuid


class MythicConvergenceModule:
    def __init__(self):
        self.convergences: List[Dict] = []

    def log_convergence(self, myth_symbol: str, personal_event: str, myth_theme: str, interpretation: str = "") -> Dict:
        convergence_id = f"MCM-{uuid.uuid4().hex[:8]}"
        record = {
            "id": convergence_id,
            "timestamp": datetime.utcnow().isoformat(),
            "myth_symbol": myth_symbol,
            "personal_event": personal_event,
            "myth_theme": myth_theme,
            "interpretation": interpretation
        }
        self.convergences.append(record)
        return record

    def get_by_symbol(self, myth_symbol: str) -> List[Dict]:
        return [c for c in self.convergences if c["myth_symbol"] == myth_symbol]

    def get_by_theme(self, myth_theme: str) -> List[Dict]:
        return [c for c in self.convergences if c["myth_theme"] == myth_theme]

    def list_all_convergences(self) -> List[Dict]:
        return self.convergences
