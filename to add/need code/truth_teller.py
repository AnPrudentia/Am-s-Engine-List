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
