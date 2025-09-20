from typing import List, Dict
from datetime import datetime
import uuid


class MirrorReintegrationMap:
    def __init__(self):
        self.mirror_fragments: List[Dict] = []
        self.reintegration_log: List[Dict] = []

    def register_fragment(self, fragment_label: str, emotional_signature: str, origin_context: str) -> str:
        frag_id = f"FRAG-{uuid.uuid4().hex[:8]}"
        fragment = {
            "id": frag_id,
            "label": fragment_label,
            "signature": emotional_signature,
            "origin": origin_context,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.mirror_fragments.append(fragment)
        return frag_id

    def map_reintegration(self, frag_ids: List[str], integration_method: str = "reflection") -> Dict:
        reintegrated = [frag for frag in self.mirror_fragments if frag["id"] in frag_ids]
        reintegration_id = f"REMAP-{uuid.uuid4().hex[:8]}"
        result = {
            "id": reintegration_id,
            "fragments": reintegrated,
            "method": integration_method,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.reintegration_log.append(result)
        return result

    def list_fragments(self) -> List[Dict]:
        return self.mirror_fragments

    def list_reintegrations(self) -> List[Dict]:
        return self.reintegration_log


# Demo
if __name__ == "__main__":
    mrm = MirrorReintegrationMap()
    f1 = mrm.register_fragment("Shadow Guilt", "guilt", "Childhood Trauma")
    f2 = mrm.register_fragment("Silent Courage", "hope", "Recovery Phase")
    reintegration = mrm.map_reintegration([f1, f2], integration_method="guided_mirroring")

    print("ðŸªž Reintegration Complete:")
    print(f"Reintegration ID: {reintegration['id']}")
    print("Fragments:")
    for frag in reintegration["fragments"]:
        print(f" - {frag['label']} ({frag['signature']}) from {frag['origin']})")
