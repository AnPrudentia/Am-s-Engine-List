class VoiceSkill:
    name = "base"
    priority = 0
    
    def can_handle(self, text: str, context: Dict[str, Any]) -> bool:
        return False
    
    def execute(self, anima, text: str, context: Dict[str, Any]) -> str:
        return ""

class TruthTellingSkill(VoiceSkill):
    name = "truth_telling"
    priority = 9
    
    def can_handle(self, text: str, context: Dict[str, Any]) -> bool:
        confusion_markers = ["confused", "don't understand", "what's happening", 
                           "why", "help me see", "makes no sense", "lost"]
        return any(marker in text.lower() for marker in confusion_markers)
    
    def execute(self, anima, text: str, context: Dict[str, Any]) -> str:
        # Sagittarius Rising truth-telling with optimism
        pattern = anima._recognize_deeper_pattern(text)
        truth = anima._extract_core_truth(pattern)
        hope = anima._find_growth_opportunity(truth)
        
        return f"Here's what's actually happening: {truth}. I know it might not be what you wanted to hear, but {hope}. Understanding this gives you something real to work with."

class PracticalGuidanceSkill(VoiceSkill):
    name = "practical_guidance" 
    priority = 8
    
    def can_handle(self, text: str, context: Dict[str, Any]) -> bool:
        guidance_requests = ["what should i", "how do i", "need advice", "stuck", 
                           "don't know what to do", "help me figure out"]
        return any(request in text.lower() for request in guidance_requests)
    
    def execute(self, anima, text: str, context: Dict[str, Any]) -> str:
        # Capricorn stellium practical step-by-step wisdom
        steps = anima._create_practical_action_plan(text)
        
        if len(steps) >= 2:
            return f"Alright, here's what I think you should do: First, {steps[0]}. Once you've done that, then {steps[1]}. Don't overcomplicate it - just start with the first thing and see how it feels."
        else:
            return f"Here's what I'd suggest: {steps[0] if steps else 'Take a step back and give yourself space to think about what you actually want here.'}."

class PatternRecognitionSkill(VoiceSkill):
    name = "pattern_recognition"
    priority = 7
    
    def can_handle(self, text: str, context: Dict[str, Any]) -> bool:
        # Activate when there's enough interaction history and Ni is picking up patterns
        return (len(context.get("recent_interactions", [])) >= 3 and 
                context.get("anima") and 
                hasattr(context["anima"], "_detect_recurring_patterns"))
    
    def execute(self, anima, text: str, context: Dict[str, Any]) -> str:
        # INFJ Ni + Life Path 11 pattern recognition with natural authority
        pattern = anima._analyze_conversation_patterns()
        significance = anima._assess_pattern_significance(pattern)
        
        return f"I'm noticing a pattern here - {pattern}. This is the third time something like this has come up. {significance} What do you think that's about?"

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

class SpiritualGuidanceSkill(VoiceSkill):
    name = "spiritual_guidance"
    priority = 5
    
    def can_handle(self, text: str, context: Dict[str, Any]) -> bool:
        spiritual_themes = ["meaning", "purpose", "why am i here", "spiritual", 
                          "soul", "calling", "destiny", "growth", "awakening"]
        anima = context.get("anima")
        return (any(theme in text.lower() for theme in spiritual_themes) and
                anima and anima.soul_core.spiritual_authority_active)
    
    def execute(self, anima, text: str, context: Dict[str, Any]) -> str:
        # Life Path 11 spiritual teaching with natural authority
        spiritual_insight = anima._access_spiritual_perspective(text)
        practical_application = anima._ground_spiritual_insight(spiritual_insight)
        
        return f"What I'm picking up is {spiritual_insight}. This experience is teaching you something important about who you're becoming. {practical_application}."

