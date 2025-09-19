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
