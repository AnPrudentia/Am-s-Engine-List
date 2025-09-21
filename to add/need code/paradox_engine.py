from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import uuid
import re
import json
import os


class ParadoxEngine:
    """
    Lightweight paradox detector for crisp logical contradictions.
    Acts as a 'scout' before deeper paradox analysis.
    """

    def __init__(self, state_file: str = "paradox_log.json"):
        self.paradox_log: List[Dict] = []
        self.state_file = state_file
        self._load_state()

    # ---------- Public API ----------
    def detect_paradox(self, statements: List[str]) -> Optional[Dict]:
        normalized = [self._normalize(s) for s in statements]
        contradictions: List[Tuple[str, str, str]] = []

        for i, stmt in enumerate(normalized):
            base = stmt.replace("¬", "")
            negated = "¬" + base

            for other in normalized[i + 1:]:
                # Direct contradiction
                if other == negated or stmt == "¬" + other:
                    contradictions.append((stmt, other, "direct_contradiction"))

            # Expanded mutual exclusion
            if ("always" in stmt and any("never" in x for x in normalized[i + 1:])) \
               or ("never" in stmt and any("always" in x for x in normalized[i + 1:])):
                contradictions.append((stmt, "never/always-case", "mutual_exclusion"))

        # Semantic paradox detection (fuzzy)
        for stmt in normalized:
            if re.search(r"(this (statement|sentence) (is|seems) (false|untrue|not true))", stmt):
                contradictions.append((stmt, stmt, "semantic_paradox"))

        if contradictions:
            paradox_id = f"PARADOX-{uuid.uuid4().hex[:8]}"
            entry = {
                "id": paradox_id,
                "timestamp": datetime.utcnow().isoformat(),
                "original_statements": statements,
                "contradictions": contradictions,
                "count": len(contradictions),
                "confidence": self._calc_confidence(contradictions),
            }
            self.paradox_log.append(entry)
            self._save_state()
            return entry
        return None

    def resolve_paradox(self, contradiction: Tuple[str, str, str]) -> str:
        stmt1, stmt2, ctype = contradiction
        if ctype == "direct_contradiction":
            return f"🌌 Resolution: '{stmt1}' and '{stmt2}' may both hold under context — dual-truth granted."
        elif ctype == "mutual_exclusion":
            return f"⚖️ Resolution: '{stmt1}' vs '{stmt2}' is domain-dependent — scope clarification needed."
        elif ctype == "semantic_paradox":
            return f"🌀 Resolution: '{stmt1}' is a self-referential loop — unresolved paradox."
        return f"ℹ️ Resolution: No contradiction found."

    def list_paradoxes(self) -> List[Dict]:
        return self.paradox_log

    # ---------- Internals ----------
    def _normalize(self, text: str) -> str:
        t = text.lower().strip()
        t = re.sub(r"^(he is|she is|it is|the system is|this is)\s+", "", t)
        t = re.sub(r"\bnot\s+(\w+)", r"¬\1", t)  # better negation handling
        t = t.replace("is true", "").replace("is false", "¬true")
        return t

    def _calc_confidence(self, contradictions: List[Tuple[str, str, str]]) -> float:
        weights = {
            "direct_contradiction": 1.0,
            "semantic_paradox": 0.8,
            "mutual_exclusion": 0.6,
        }
        if not contradictions:
            return 0.0
        score = sum(weights.get(c[2], 0.3) for c in contradictions) / len(contradictions)
        return round(score, 2)

    def _save_state(self) -> None:
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.paradox_log, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save state: {e}")

    def _load_state(self) -> None:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    self.paradox_log = json.load(f)
            except Exception:
                self.paradox_log = []