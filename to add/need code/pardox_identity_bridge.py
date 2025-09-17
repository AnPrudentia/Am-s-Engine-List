from typing import Dict, List
from datetime import datetime
import uuid

class ParadoxIdentityBridge:
    def __init__(self):
        self.bridges: List[Dict] = []

    def construct_bridge(self, identities: List[str], paradox: str, context: str = "") -> Dict:
        bridge_id = f"PIB-{uuid.uuid4().hex[:8]}"
        bridge_data = {
            "id": bridge_id,
            "timestamp": datetime.utcnow().isoformat(),
            "identities": identities,
            "paradox": paradox,
            "context": context,
            "resolved_state": self.resolve_paradox(identities, paradox)
        }
        self.bridges.append(bridge_data)
        return bridge_data

    def resolve_paradox(self, identities: List[str], paradox: str) -> str:
        if len(identities) < 2:
            return "Insufficient identity facets for bridge formation."

        synthesis = f"Paradox reconciled through identity synthesis: {' + '.join(identities)} â†’ {paradox} reframed."
        return synthesis

    def list_bridges(self) -> List[Dict]:
        return self.bridges
