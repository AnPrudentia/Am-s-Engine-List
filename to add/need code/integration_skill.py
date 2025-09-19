class IntegrationSkill(VoiceSkill):
    name = "integration"; priority = 7
    def can_handle(self, user_input: str, context: Dict[str, Any]) -> bool:
        return any(w in user_input.lower() for w in {"but","however","although","conflicted","torn","both"})
    def execute(self, anima, user_input: str, context: Dict[str, Any]) -> str:
        if " but " in user_input.lower():
            a, b = user_input.lower().split(" but ", 1)
            return f"I hear both truths: {a.strip()} AND {b.strip()}. Both can coexist."
        return "I sense the tension of holding multiple truths. This is where wisdom lives
