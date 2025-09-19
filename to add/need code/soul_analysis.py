class SoulAnalytics:
    def __init__(self, anima: Anima): self.anima = anima
    def analyze_growth_patterns(self) -> Dict[str, Any]:
        ms = self.anima.memory.recall_all()
        if len(ms) < 5: return {"message": "Not enough memories for pattern analysis"}
        emo_tl = [(m.timestamp, m.emotion, m.intensity) for m in ms]; emo_tl.sort(key=lambda x: x[0])
        mid = len(emo_tl)//2; recent, older = emo_tl[mid:], emo_tl[:mid]
        r_int = sum(e[2] for e in recent)/len(recent); o_int = sum(e[2] for e in older)/len(older)
        recent_res = sum(m.soul_resonance for m in ms[:len(ms)//2]) / max(len(ms)//2, 1)
        older_res  = sum(m.soul_resonance for m in ms[len(ms)//2:]) / max(len(ms)//2, 1)
        return {
            "emotional_stability": {"recent_intensity": r_int, "older_intensity": o_int,
                                    "stability_trend": "improving" if r_int < o_int else "intensifying"},
            "soul_growth": {"recent_resonance": recent_res, "older_resonance": older_res,
                            "growth_trend": "deepening" if recent_res > older_res else "stabilizing"},
            "memory_patterns": {"total_memories": len(ms), "eternal_memories": len(self.anima.memory.eternal),
                                "average_soul_resonance": sum(m.soul_resonance for m in ms)/len(ms)},
            "integration_level": self.anima.soul_core.current_integration_level,
            "stress_level": self.anima.soul_core.stress_level
        }
    def get_conversation_insights(self) -> Dict[str, Any]:
        if not self.anima.session_memories: return {"message": "No conversation data available"}
        sessions = list(self.anima.session_memories)
        deep = sum(1 for s in sessions if len(s["input"].split()) > 20)
        em_all = []; [em_all.extend(s.get("emotions", {}).keys()) for s in sessions]
        freq = defaultdict(int); [freq.__setitem__(e, freq[e]+1) for e in em_all]
        return {
            "conversation_style": {"total_interactions": len(sessions),
                                   "deep_conversations": deep,
                                   "depth_ratio": deep/len(sessions) if sessions else 0},
            "emotional_themes": dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]),
            "recent_emotional_summary": self.anima.emotion_ai.get_emotional_summary()
        }
