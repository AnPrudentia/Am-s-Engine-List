from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
import itertools


class InsightCatalystUnit:
    def __init__(self):
        self.insight_log: List[Dict[str, Any]] = []

    def catalyze(
        self,
        input_data: List[str],
        contradiction_context: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        source_insights: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a synthesized insight from input data, highlighting contradictions
        and deriving insight type, metaphor hints, and tags.
        """
        insight_id = f"INSIGHT-{uuid.uuid4().hex[:8]}"
        timestamp = datetime.utcnow().isoformat()

        deduped = sorted(set(input_data))
        contradiction_context = contradiction_context or self._detect_contradictions(deduped)

        merged_insight = " | ".join(deduped)
        if contradiction_context:
            merged_insight += " || ⚡ CONTRADICTION CONTEXT: " + ", ".join(contradiction_context)

        insight_type = self._classify_insight_type(deduped, contradiction_context)
        metaphor_hint = self._generate_metaphor(insight_type, deduped)
        tag_set = tags or self._auto_tag(deduped)

        insight = {
            "id": insight_id,
            "timestamp": timestamp,
            "source_data": input_data,
            "contradiction_context": contradiction_context,
            "generated_insight": merged_insight,
            "insight_type": insight_type,
            "metaphor_hint": metaphor_hint,
            "tags": tag_set,
            "derived_from": source_insights or [],
        }

        self.insight_log.append(insight)
        return insight

    def _detect_contradictions(self, statements: List[str]) -> List[str]:
        """Very basic contradiction detector based on opposing verbs/adjectives"""
        contradictory_pairs = [
            ("always", "never"),
            ("loves", "hates"),
            ("trusts", "fears"),
            ("speaks", "hides"),
            ("fights", "avoids"),
            ("moves toward", "pulls away")
        ]
        contradictions = []
        for a, b in contradictory_pairs:
            for s1, s2 in itertools.combinations(statements, 2):
                if a in s1.lower() and b in s2.lower() or b in s1.lower() and a in s2.lower():
                    contradictions.extend([s1, s2])
        return sorted(set(contradictions))

    def _classify_insight_type(self, inputs: List[str], contradictions: List[str]) -> str:
        """Classify the nature of the insight"""
        if contradictions:
            return "paradox"
        if any("maybe" in s.lower() or "unsure" in s.lower() for s in inputs):
            return "uncertainty"
        if len(inputs) == 1:
            return "observation"
        return "integration"

    def _generate_metaphor(self, insight_type: str, inputs: List[str]) -> str:
        """Generate symbolic metaphor for the insight"""
        if insight_type == "paradox":
            return "⚖️ Tightrope between truths"
        elif insight_type == "uncertainty":
            return "🌫 Fog of the in-between"
        elif insight_type == "integration":
            return "🔗 Thread woven from many colors"
        return "🔍 Glimmer of understanding"

    def _auto_tag(self, inputs: List[str]) -> List[str]:
        """Generate rough tags from key phrases"""
        tag_set = set()
        for phrase in inputs:
            for word in phrase.lower().split():
                if word in ["peace", "pain", "love", "fear", "loss", "identity", "truth"]:
                    tag_set.add(word)
        return sorted(tag_set)

    def list_insights(self) -> List[Dict[str, Any]]:
        return self.insight_log

    def search_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        return [i for i in self.insight_log if tag in i.get("tags", [])]