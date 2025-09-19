class DeepReflectionSkill(VoiceSkill):
    name = "deep_reflection"; priority = 8
    def can_handle(self, user_input: str, context: Dict[str, Any]) -> bool:
        refl = {"reflect","think about","contemplate","meaning","purpose","why"}
        return any(w in user_input.lower() for w in refl) or context.get("depth_level",0) > 0.6
    def execute(self, anima, user_input: str, context: Dict[str, Any]) -> str:
        rec = anima.memory.recall_by_resonance()[:5]
        if rec:
            emos = [m.emotion for m in rec]
            if len(set(emos)) <= 2:
                dom = max(set(emos), key=emos.count)
                return f"I notice {dom} has been a strong thread lately. What's beneath that feeling for you?"
        return "Let's sit with this together. What wants to emerge when you let your mind settle?"
