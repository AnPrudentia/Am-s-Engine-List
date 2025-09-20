from __future__ import annotations
from typing import Dict, Any, List
from collections import defaultdict
from statistics import mean
import json


class SoulAnalytics:
    """
    Reflective analytics layer for Anima.
    Tracks growth, stability, and conversational/emotional trends.
    """

    def __init__(self, anima: Any):
        self.anima = anima

    # ---------- Growth Patterns ----------
    def analyze_growth_patterns(self) -> Dict[str, Any]:
        ms = self.anima.memory.recall_all()
        if len(ms) < 5:
            return {"message": "Not enough memories for pattern analysis"}

        # Build emotional timeline
        emo_tl = [(m.timestamp, getattr(m, "emotion", None), getattr(m, "intensity", 0.0)) for m in ms]
        emo_tl.sort(key=lambda x: x[0])  # chronological order

        # Split into older vs recent half
        mid = len(emo_tl) // 2
        recent, older = emo_tl[mid:], emo_tl[:mid]

        r_int = mean([e[2] for e in recent]) if recent else 0.0
        o_int = mean([e[2] for e in older]) if older else 0.0

        # Soul resonance tracking
        half = len(ms) // 2 or 1
        recent_res = mean([getattr(m, "soul_resonance", 0.0) for m in ms[:half]])
        older_res = mean([getattr(m, "soul_resonance", 0.0) for m in ms[half:]])

        return {
            "emotional_stability": {
                "recent_intensity": r_int,
                "older_intensity": o_int,
                "stability_trend": self._trend_label(r_int, o_int, "intensity"),
            },
            "soul_growth": {
                "recent_resonance": recent_res,
                "older_resonance": older_res,
                "growth_trend": self._trend_label(recent_res, older_res, "resonance"),
            },
            "memory_patterns": {
                "total_memories": len(ms),
                "eternal_memories": len(getattr(self.anima.memory, "eternal", [])),
                "average_soul_resonance": mean([getattr(m, "soul_resonance", 0.0) for m in ms]),
            },
            "integration_level": getattr(self.anima.soul_core, "current_integration_level", None),
            "stress_level": getattr(self.anima.soul_core, "stress_level", None),
        }

    # ---------- Conversation Insights ----------
    def get_conversation_insights(self) -> Dict[str, Any]:
        sessions = getattr(self.anima, "session_memories", [])
        if not sessions:
            return {"message": "No conversation data available"}

        deep = sum(1 for s in sessions if len(s.get("input", "").split()) > 20)

        # Collect emotional themes
        freq = defaultdict(int)
        for s in sessions:
            for emo in s.get("emotions", {}).keys():
                freq[emo] += 1

        top_themes = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5])

        return {
            "conversation_style": {
                "total_interactions": len(sessions),
                "deep_conversations": deep,
                "depth_ratio": round(deep / len(sessions), 2) if sessions else 0,
            },
            "emotional_themes": top_themes,
            "recent_emotional_summary": getattr(self.anima.emotion_ai, "get_emotional_summary", lambda: "N/A")(),
        }

    # ---------- Helpers ----------
    def _trend_label(self, recent: float, older: float, kind: str) -> str:
        if abs(recent - older) < 0.05:
            return f"{kind} stable"
        return f"{kind} improving" if recent < older else f"{kind} intensifying"

    def to_dict(self) -> Dict[str, Any]:
        """Export both growth and conversation analytics."""
        return {
            "growth_patterns": self.analyze_growth_patterns(),
            "conversation_insights": self.get_conversation_insights(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)