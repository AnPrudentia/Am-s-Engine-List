class VoiceSkill:
    name = "base"; priority = 0
    def can_handle(self, user_input: str, context: Dict[str, Any]) -> bool: return False
    def execute(self, anima, user_input: str, context: Dict[str, Any]) -> str: return ""
