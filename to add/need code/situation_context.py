from typing import List, Dict, Optional
from datetime import datetime
import re
import uuid


class SituationalContext:
    def __init__(self, trigger: str, context_id: str, label: str, description: str):
        self.trigger = trigger
        self.context_id = context_id
        self.label = label
        self.description = description


class SituationalContextEngine:
    def __init__(self):
        self.context_patterns: List[SituationalContext] = [
            SituationalContext("i don't know what to choose", "choice_paralysis", "Choice Paralysis",
                               "When no option feels safe enough to commit to."),
            SituationalContext("i keep messing up", "guilt_loop", "Internalized Guilt Loop",
                               "A pattern where self-blame becomes a safety mechanism."),
            SituationalContext("i feel split", "identity_dissonance", "Identity Dissonance",
                               "When multiple internal roles or values are at odds."),
            SituationalContext("they say they love me, but", "conditional_acceptance", "Conditional Love Conflict",
                               "When affection is offered with strings attached."),
            SituationalContext("no one sees me", "invisibility_wound", "Invisibility Wound",
                               "A recurring feeling of being unseen, unheard, or emotionally erased."),
            SituationalContext("i don't know who i am", "identity_void", "Identity Void",
                               "A loss of self-definition during transitions or emotional fragmentation."),
            SituationalContext("i feel everything and nothing", "emotional_flood", "Dissociative Flooding",
                               "Overwhelm that leads to emotional shutdown or numbness."),
            SituationalContext("i want to disappear", "escapist_collapse", "Escapist Collapse",
                               "A desire to escape visibility or responsibility when pressure peaks."),
        ]

    def detect(self, text: str) -> List[Dict]:
        matches = []
        normalized = text.lower()
        for ctx in self.context_patterns:
            if ctx.trigger in normalized:
                matches.append({
                    "context_id": ctx.context_id,
                    "label": ctx.label,
                    "description": ctx.description
                })
        return matches or [{
            "context_id": "ambiguous_context",
            "label": "Ambiguous Situation",
            "description": "Context unclear. May need more introspective detail."
        }]

    def annotate(self, text: str) -> Dict:
        contexts = self.detect(text)
        return {
            "input": text,
            "contexts_detected": contexts,
            "analysis_id": f"CTX-{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.utcnow().isoformat()
        }