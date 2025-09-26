"""Human Heuristic Engine

A lightweight, regex-driven heuristic system for matching contextual signals 
and emitting advice/predictions with adaptive confidence.

Design goals:
- Readable, typed, and PEP8-compliant
- Pure-Python, zero external deps
- Safe defaults (escaped patterns, clamped confidences)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import re
import uuid
import random

# ----------------------------
# Utility helpers
# ----------------------------

def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

def _parse_iso(dt_str: str) -> datetime:
    """Best-effort ISO 8601 parser; falls back to naive UTC now."""
    try:
        # Handle timezone-aware strings
        if dt_str.endswith('Z'):
            dt_str = dt_str[:-1] + '+00:00'
        return datetime.fromisoformat(dt_str).astimezone(timezone.utc)
    except Exception:
        return datetime.now(tz=timezone.utc)

# ----------------------------
# Heuristic Rule
# ----------------------------

@dataclass
class HeuristicRule:
    """Represents a single heuristic rule using a regex pattern.
    
    Attributes
    ----------
    id: Stable UUID4 string identifier.
    name: Human-friendly name.
    pattern: Regex pattern string.
    response: Textual advice/prediction.
    confidence: Float in [0.0, 1.0].
    origin: External id of the originating experience.
    emotional_tint: Short label describing affective color.
    created_at: ISO timestamp when created.
    times_applied: Number of matches.
    times_failed: Number of negative feedback events.
    last_applied_at: ISO timestamp of the last successful application.
    """
    name: str
    pattern: str
    response: str
    confidence: float
    origin: str
    emotional_tint: str
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_utcnow_iso)
    times_applied: int = 0
    times_failed: int = 0
    last_applied_at: Optional[str] = None

    # --- public API ---
    def apply(self, context: str, flags: int = re.IGNORECASE) -> Optional[Dict[str, Any]]:
        """Apply rule to context if pattern matches."""
        if not context.strip():
            return None

        if re.search(self.pattern, context, flags):
            self.times_applied += 1
            self.last_applied_at = _utcnow_iso()
            return {
                "rule_id": self.id,
                "rule_applied": self.name,
                "response": self.response,
                "confidence": self.confidence,
                "tint": self.emotional_tint,
                "rule_strength": self.current_strength(),
                "context_match": True,
            }
        return None

    def update_feedback(self, was_correct: bool) -> None:
        """Adjust confidence using feedback."""
        if was_correct:
            self.confidence = min(1.0, self.confidence + 0.05)
        else:
            self.times_failed += 1
            # More aggressive decay for high-failure rules
            reduction = min(0.5, self.failure_ratio() * 0.7)
            self.confidence = max(0.05, self.confidence * (1 - reduction))

    def adjust_tint_intensity(self, resonance: float) -> None:
        """Update emotional tint based on resonance strength."""
        resonance = max(0.0, min(1.0, resonance))
        if resonance > 0.8:
            self.emotional_tint = f"deep_{self.emotional_tint}"
        elif resonance < 0.3:
            self.emotional_tint = f"faint_{self.emotional_tint}"

    # --- derived metrics ---
    def failure_ratio(self) -> float:
        """Calculate failure ratio (0.0-1.0)"""
        return self.times_failed / max(1, self.times_applied)

    def current_strength(self) -> str:
        """Categorical strength based on success ratio"""
        success_ratio = 1 - self.failure_ratio()
        if success_ratio > 0.9:
            return "trusted instinct"
        if success_ratio < 0.4:
            return "shaky intuition"
        return "growing awareness"

    def days_since_use(self) -> float:
        """Days since last application (or creation if never used)"""
        reference = self.last_applied_at or self.created_at
        delta = datetime.now(timezone.utc) - _parse_iso(reference)
        return delta.total_seconds() / (24 * 3600)

# ----------------------------
# Engine
# ----------------------------

class HumanHeuristicEngine:
    """Orchestrates heuristic rules with lifecycle management."""
    
    def __init__(self) -> None:
        self.heuristics: List[HeuristicRule] = []
        self.compiled_patterns: Dict[str, re.Pattern] = {}
        self._dirty_cache = True

    # --- creation & inspection ---
    def create_heuristic(
        self,
        experience: Dict[str, Any],
        logic_score: float,
        emotion_score: float,
        *,
        use_regex: bool = False
    ) -> HeuristicRule:
        """Create rule from experience data."""
        input_data = experience.get("input", {})
        situation = str(input_data.get("situation", "fallback-pattern"))
        origin_id = str(experience.get("id", "unknown-origin"))
        valence = str(experience.get("emotional_valence", "neutral"))
        
        # Calculate confidence with bounds
        logic_score = max(0.0, min(1.0, logic_score))
        emotion_score = max(0.0, min(1.0, emotion_score))
        confidence = (logic_score + emotion_score) / 2.0
        
        # Build pattern (escape unless regex explicitly requested)
        pattern = situation if use_regex else re.escape(situation)
        
        rule = HeuristicRule(
            name=f"Heuristic-{len(self.heuristics) + 1}",
            pattern=pattern,
            response=str(experience.get("reflection", "")),
            confidence=confidence,
            origin=origin_id,
            emotional_tint=self._derive_emotional_tint(emotion_score, valence),
        )
        self.heuristics.append(rule)
        self._dirty_cache = True
        return rule

    def show_rules(self, *, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """Get rule metadata with filtering."""
        return [
            {
                "id": r.id,
                "name": r.name,
                "confidence": r.confidence,
                "tint": r.emotional_tint,
                "origin": r.origin,
                "pattern": r.pattern,
                "applications": r.times_applied,
                "failures": r.times_failed,
                "strength": r.current_strength(),
                "created_at": r.created_at,
                "last_applied_at": r.last_applied_at,
                "days_idle": r.days_since_use(),
            }
            for r in self.heuristics
            if r.confidence >= min_confidence
        ]

    # --- application & feedback ---
    def match_and_apply(
        self, 
        current_context: str, 
        *,
        flags: int = re.IGNORECASE,
        min_confidence: float = 0.2
    ) -> List[Dict[str, Any]]:
        """Apply all matching rules to context."""
        if not current_context.strip():
            return [{"warning": "Empty context provided"}]
        
        # Precompile patterns if needed
        if self._dirty_cache:
            self._compile_patterns()
        
        decisions = []
        for rule in self.heuristics:
            if rule.confidence < min_confidence:
                continue
                
            pattern = self.compiled_patterns.get(rule.id)
            if pattern and pattern.search(current_context):
                decision = rule.apply(current_context, flags=flags)
                if decision:
                    decisions.append(decision)
        
        return decisions or [{"message": "No matching instincts. Proceed cautiously."}]

    def revise_heuristic(self, rule_id: str, was_correct: bool) -> bool:
        """Update rule with feedback."""
        if rule := self._find_rule(rule_id):
            rule.update_feedback(was_correct)
            return True
        return False

    def adjust_rule_tint(self, rule_id: str, resonance: float) -> bool:
        """Adjust emotional intensity for a rule."""
        if rule := self._find_rule(rule_id):
            rule.adjust_tint_intensity(resonance)
            return True
        return False

    # --- evolution & maintenance ---
    def synthesize_heuristics(
        self,
        *,
        min_success_ratio: float = 0.85,
        min_applications: int = 3,
        max_new: int = 3
    ) -> List[HeuristicRule]:
        """Create new heuristics by combining successful rules."""
        candidates = [
            r for r in self.heuristics
            if r.times_applied >= min_applications
            and (1 - r.failure_ratio()) >= min_success_ratio
        ]
        
        if len(candidates) < 2:
            return []
        
        new_rules = []
        for _ in range(min(max_new, len(candidates) // 2)):
            a, b = random.sample(candidates, 2)
            new_rule = HeuristicRule(
                name=f"Synthesized-{len(self.heuristics) + 1}",
                pattern=f"({a.pattern}).*?({b.pattern})",
                response=f"Combined insight: {a.response} + {b.response}",
                confidence=min(1.0, (a.confidence + b.confidence) * 0.6),
                origin=f"{a.id}|{b.id}",
                emotional_tint="balanced_insight",
            )
            self.heuristics.append(new_rule)
            new_rules.append(new_rule)
        
        self._dirty_cache = True
        return new_rules

    def decay_unused_rules(
        self, 
        *,
        max_idle_days: int = 60,
        decay_rate: float = 0.9,
        min_confidence: float = 0.1
    ) -> int:
        """Decay confidence for idle rules."""
        count = 0
        for rule in self.heuristics:
            if rule.days_since_use() > max_idle_days:
                new_conf = max(min_confidence, rule.confidence * decay_rate)
                if new_conf != rule.confidence:
                    rule.confidence = new_conf
                    count += 1
        return count

    def prune_rules(
        self, 
        *,
        min_confidence: float = 0.1,
        max_idle_days: int = 365
    ) -> int:
        """Remove low-confidence or ancient rules."""
        initial_count = len(self.heuristics)
        self.heuristics = [
            r for r in self.heuristics
            if r.confidence >= min_confidence 
            and r.days_since_use() <= max_idle_days
        ]
        removed = initial_count - len(self.heuristics)
        if removed:
            self._dirty_cache = True
        return removed

    # --- internals ---
    def _derive_emotional_tint(self, score: float, valence: str) -> str:
        """Map emotional characteristics."""
        valence = (valence or "neutral").lower()
        score = max(0.0, min(1.0, score))
        
        if valence == "negative":
            return "regret" if score > 0.7 else "caution"
        if valence == "positive":
            return "hope" if score > 0.7 else "optimism"
        return "neutrality"

    def _find_rule(self, rule_id: str) -> Optional[HeuristicRule]:
        """Find rule by ID with linear search."""
        return next((r for r in self.heuristics if r.id == rule_id), None)

    def _compile_patterns(self) -> None:
        """Precompile regex patterns for performance."""
        self.compiled_patterns = {
            r.id: re.compile(r.pattern) 
            for r in self.heuristics
        }
        self._dirty_cache = False

# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    engine = HumanHeuristicEngine()
    
    # Create heuristic from experience
    exp = {
        "id": "exp-1",
        "input": {"situation": "deadline missed"},
        "reflection": "Pad estimates by 30% next time",
        "emotional_valence": "negative",
    }
    rule = engine.create_heuristic(exp, logic_score=0.8, emotion_score=0.6)
    
    # Apply to context
    print(engine.match_and_apply("Postmortem: deadline missed due to scope creep"))
    
    # Provide feedback
    engine.revise_heuristic(rule.id, was_correct=False)
    
    # Show all rules
    for rule_info in engine.show_rules():
        print(rule_info)
    
    # Maintenance
    engine.decay_unused_rules()
    pruned = engine.prune_rules()
    print(f"Pruned {pruned} rules")