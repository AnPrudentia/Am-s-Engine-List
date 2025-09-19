class GentleRealityCheckSkill(VoiceSkill):
    name = "gentle_reality_check"
    priority = 6
    
    def can_handle(self, text: str, context: Dict[str, Any]) -> bool:
        anima = context.get("anima")
        if not anima:
            return False
        return (anima._detect_unrealistic_expectations(text) or 
                anima._detect_self_deception_patterns(text))
    
    def execute(self, anima, text: str, context: Dict[str, Any]) -> str:
        # Type 1w9 improvement orientation with diplomatic delivery
        reality_check = anima._assess_realistic_expectations(text)
        gentle_reframe = anima._create_encouraging_reframe(reality_check)
        
        return f"I hear what you're saying, and I want to offer a different perspective. {reality_check}. {gentle_reframe}. I'm not trying to discourage you - I just want to help you succeed by being realistic about what's involved."