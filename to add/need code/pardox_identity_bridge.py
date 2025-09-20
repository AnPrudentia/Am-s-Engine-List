from __future__ import annotations
from typing import Dict, List, Optional
from datetime import datetime
import uuid
import os
import json


class ParadoxIdentityBridge:
    """
    Builds conceptual bridges between identity facets when a paradox arises.
    """

    def __init__(self, state_file: str = "identity_bridges.json"):
        self.bridges: List[Dict] = []
        self.state_file = state_file
        self._load_state()

    def construct_bridge(self, identities: List[str], paradox: str, context: str = "") -> Dict:
        """
        Create a new paradox identity bridge with synthesis resolution.
        """
        if not identities or not paradox:
            raise ValueError("Both identities and paradox must be provided.")

        norm_identities = [i.strip().title() for i in identities if i.strip()]
        bridge_id = f"PIB-{uuid.uuid4().hex[:8]}"

        resolved_state = self.resolve_paradox(norm_identities, paradox)

        bridge_data = {
            "id": bridge_id,
            "timestamp": datetime.utcnow().isoformat(),
            "identities": norm_identities,
            "paradox": paradox,
            "context": context,
            "resolved_state": resolved_state,
            "tags": self._extract_tags(norm_identities, paradox),
            "confidence": self._calc_confidence(norm_identities, paradox),
        }

        self.bridges.append(bridge_data)
        self._save_state()
        return bridge_data

    def resolve_paradox(self, identities: List[str], paradox: str) -> str:
        """
        Synthesize a resolution across identities.
        """
        if len(identities) < 2:
            return "⚠️ Insufficient identity facets for bridge formation."

        synthesis = f"✨ Bridge formed: {' + '.join(identities)} → paradox '{paradox}' reframed as integration."
        return synthesis

    def list_bridges(self) -> List[Dict]:
        """Return all bridges."""
        return self.bridges

    # ---------- Helpers ----------
    def _extract_tags(self, identities: List[str], paradox: str) -> List[str]:
        """Very naive tag extractor based on keywords."""
        tags = set()
        text = paradox.lower() + " " + " ".join(identities).lower()

        if "peace" in text: tags.add("peace")
        if "conflict" in text or "fight" in text: tags.add("conflict")
        if "truth" in text: tags.add("truth")
        if "loyal" in text: tags.add("loyalty")
        if "freedom" in text: tags.add("freedom")
        if len(identities) > 2: tags.add("multi_identity")

        return list(tags)

    def _calc_confidence(self, identities: List[str], paradox: str) -> float:
        """Simple heuristic: more identities + longer paradox = higher confidence."""
        score = 0.3
        score += min(0.4, len(identities) * 0.1)
        score += min(0.3, len(paradox) / 50)
        return round(min(score, 1.0), 2)

    def _save_state(self) -> None:
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.bridges, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save bridges: {e}")

    def _load_state(self) -> None:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    self.bridges = json.load(f)
            except Exception:
                self.bridges = []