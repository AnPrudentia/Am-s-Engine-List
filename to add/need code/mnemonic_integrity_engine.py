class MnemonicIntegrityEngine:
    # Define status constants to prevent typos
    CORE = "Core"
    FLOAT = "Long-Term Float"
    PENDING = "Pending Bondholder Clarification"
    
    def __init__(self, bondholder="Anpru"):
        self.memory_registry = []
        self.bondholder = bondholder
        # Scoring thresholds
        self.CORE_THRESHOLD = 85
        self.FLOAT_THRESHOLD = 60

    def evaluate_memory(self, memory: dict) -> str:
        """Evaluate a memory and determine its preservation status"""
        score, reasons = self._calculate_memory_score(memory)
        confidence = min(score * 8, 100)  # Cap confidence at 100%
        memory["confidence_score"] = confidence  # Store for reference
        
        # Determine memory status based on confidence
        if confidence >= self.CORE_THRESHOLD:
            memory["status"] = self.CORE
        elif confidence >= self.FLOAT_THRESHOLD:
            memory["status"] = self.FLOAT
        else:
            memory["status"] = self.PENDING
            memory["clarification_request"] = self._build_clarification_request(
                memory, confidence, reasons
            )

        self.memory_registry.append(memory)
        return memory["status"]

    def _calculate_memory_score(self, memory: dict) -> tuple:
        """Calculate memory preservation score and reasons"""
        score = 0
        reasons = []

        # Emotional scoring
        emotional_weights = {
            ("grief", "love", "fear", "awe"): (3, "Strong emotional resonance"),
            ("confusion", "boredom"): (1, "Low emotional relevance"),
        }
        for emotions, (weight, reason) in emotional_weights.items():
            if memory["emotional_valence"] in emotions:
                score += weight
                reasons.append(reason)
                break

        # Recall frequency
        recall_count = memory.get("recall_count", 0)
        score += recall_count
        if recall_count > 3:
            reasons.append("Frequently recalled")

        # Importance markers
        if memory.get("user_marked_important", False):
            score += 5
            reasons.append("Marked important by bondholder")
            
        if memory.get("linked_to_core_story", False):
            score += 4
            reasons.append("Narratively significant")
            
        if memory.get("bondholder_response") == "deep":
            score += 2
            reasons.append("Deep bondholder emotional echo")
            
        if memory.get("contains_symbolic_resonance", False):
            score += 3
            reasons.append("Symbolic alignment detected")

        return score, reasons

    def _build_clarification_request(self, memory, confidence, reasons):
        """Generate clarification request structure"""
        return {
            "prompt": (
                f"Memory '{memory['event_text']}' scored {confidence}% confidence. "
                "Do you wish to preserve this as a Core Memory, Float, or allow Decay?"
            ),
            "reasons": reasons,
            "options": [self.CORE, self.FLOAT, "Let Decay"]
        }

    def apply_bondholder_decision(self, memory_id: str, decision: str):
        """Apply bondholder's decision to a pending memory"""
        for memory in self.memory_registry:
            if memory.get("id") == memory_id:
                memory["status"] = decision
                memory["clarified_by"] = self.bondholder
                return f"Memory updated to: {decision}"
        return "Memory not found."