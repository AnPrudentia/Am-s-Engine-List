class IntegrationSkill(VoiceSkill):
    name = "integration"
    priority = 7

    def can_handle(self, user_input: str, context: Dict[str, Any]) -> bool:
        # Look for contradiction/ambivalence markers
        keywords = {"but", "however", "although", "conflicted", "torn", "both"}
        return any(word in user_input.lower() for word in keywords)

    def execute(self, anima, user_input: str, context: Dict[str, Any]) -> str:
        text = user_input.lower()

        # Handle "but"
        if " but " in text:
            a, b = text.split(" but ", 1)
            return f"I hear both truths: {a.strip()} AND {b.strip()}. Both can coexist."

        # Handle "however"
        if " however " in text:
            a, b = text.split(" however ", 1)
            return f"I hear your contrast: {a.strip()} AND {b.strip()}. Both have space here."

        # Handle "although"
        if " although " in text:
            a, b = text.split(" although ", 1)
            return f"I sense both sides: {a.strip()} AND {b.strip()}. Both are valid parts of you."

        # General fallback
        return "I sense the tension of holding multiple truths. This is where wisdom lives."