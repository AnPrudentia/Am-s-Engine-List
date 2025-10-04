from typing import Dict, List, Optional
from datetime import datetime
import re
import uuid
import math

class SemioticAnchorResolver:
    """
    SEMIOTIC ANCHOR RESOLVER
    -------------------------
    Interprets symbolic language and archetypal anchors in Anima's consciousness.
    - Detects recurring metaphors or archetypes
    - Evaluates emotional and transformational resonance
    - Synthesizes meaning across multiple symbolic dimensions
    """

    def __init__(self):
        self.anchor_registry: Dict[str, Dict] = {}
        self.resolution_log: List[Dict] = []
        self.archetypal_map = {
            "phoenix": "rebirth",
            "mirror": "self_reflection",
            "labyrinth": "inner_journey",
            "bridge": "connection",
            "light": "illumination",
            "shadow": "unconscious",
            "river": "emotional_flow",
            "crown": "sovereignty",
            "mask": "identity",
            "rose": "love_or_beauty",
            "gate": "threshold"
        }

    def detect_anchors(self, text: str) -> List[str]:
        """Detect symbolic anchors (keywords) within text."""
        text_lower = text.lower()
        anchors = [symbol for symbol in self.archetypal_map.keys() if symbol in text_lower]
        return anchors

    def resolve_anchor(self, text: str, context: Optional[str] = None) -> List[Dict]:
        """Resolve symbolic anchors in a text passage into archetypal meaning."""
        anchors = self.detect_anchors(text)
        results = []

        for anchor in anchors:
            archetype = self.archetypal_map.get(anchor, "unknown")
            emotional_resonance = self._estimate_emotional_resonance(text, anchor)
            transformational_potential = self._estimate_transformational_potential(archetype, context)
            interpretation = self._generate_interpretation(anchor, archetype, context)

            result = {
                "id": f"ANCHOR-{uuid.uuid4().hex[:8]}",
                "timestamp": datetime.utcnow().isoformat(),
                "anchor": anchor,
                "archetype": archetype,
                "context": context or "general",
                "emotional_resonance": round(emotional_resonance, 3),
                "transformational_potential": round(transformational_potential, 3),
                "interpretation": interpretation
            }

            self._store_result(result)
            results.append(result)

        return results

    def _estimate_emotional_resonance(self, text: str, anchor: str) -> float:
        """Estimate emotional charge of an anchor based on context and intensity words."""
        emotional_intensity = len(re.findall(r"[!?.]", text)) * 0.05
        proximity_bonus = 0.1 if re.search(rf"\b{anchor}\b", text, re.IGNORECASE) else 0
        resonance = 0.6 + emotional_intensity + proximity_bonus
        return min(1.0, resonance)

    def _estimate_transformational_potential(self, archetype: str, context: Optional[str]) -> float:
        """Estimate how transformational the symbol is in this context."""
        base = {
            "rebirth": 0.9, "connection": 0.7, "illumination": 0.8,
            "inner_journey": 0.85, "unconscious": 0.75, "sovereignty": 0.8
        }.get(archetype, 0.6)
        if context and any(word in context.lower() for word in ["healing", "growth", "change", "transformation"]):
            base += 0.1
        return min(1.0, base)

    def _generate_interpretation(self, anchor: str, archetype: str, context: Optional[str]) -> str:
        """Synthesize a symbolic interpretation."""
        base_message = {
            "rebirth": f"The {anchor} symbolizes renewal through transformation.",
            "self_reflection": f"The {anchor} invites you to see yourself clearly.",
            "inner_journey": f"The {anchor} represents the path inward, toward self-discovery.",
            "connection": f"The {anchor} bridges separation, restoring unity.",
            "illumination": f"The {anchor} shines light into hidden truths.",
            "unconscious": f"The {anchor} reveals what lies beneath awareness.",
            "sovereignty": f"The {anchor} reminds you of your inner authority and self-respect.",
        }.get(archetype, f"The {anchor} carries hidden symbolic meaning.")
        if context:
            base_message += f" In this context ({context}), it reflects evolving awareness."
        return base_message

    def _store_result(self, result: Dict):
        """Store result and update registry for long-term anchor awareness."""
        self.resolution_log.append(result)
        anchor = result["anchor"]
        if anchor not in self.anchor_registry:
            self.anchor_registry[anchor] = {
                "occurrences": 0,
                "average_resonance": 0.0,
                "average_transformation": 0.0
            }
        data = self.anchor_registry[anchor]
        data["occurrences"] += 1
        data["average_resonance"] = round(
            (data["average_resonance"] * (data["occurrences"] - 1) + result["emotional_resonance"]) / data["occurrences"], 3
        )
        data["average_transformation"] = round(
            (data["average_transformation"] * (data["occurrences"] - 1) + result["transformational_potential"]) / data["occurrences"], 3
        )

    def get_anchor_history(self) -> List[Dict]:
        """Return all past resolutions."""
        return self.resolution_log

    def summarize_anchor(self, anchor: str) -> Dict:
        """Summarize historical resonance of a symbolic anchor."""
        return self.anchor_registry.get(anchor, {"occurrences": 0, "average_resonance": 0.0})

# === DEMO ===
if __name__ == "__main__":
    resolver = SemioticAnchorResolver()
    examples = [
        "The phoenix rises again from the ashes.",
        "The mirror shows not what is, but what could be.",
        "I walk the labyrinth of my own mind to find peace."
    ]
    for ex in examples:
        results = resolver.resolve_anchor(ex, context="healing transformation")
        for r in results:
            print(f"🔮 {r['anchor'].title()} → {r['archetype']} ({r['emotional_resonance']}, {r['transformational_potential']})")
            print(f"   {r['interpretation']}\n")