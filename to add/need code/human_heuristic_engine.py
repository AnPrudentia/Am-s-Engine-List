src/human_heuristic_engine.py

"""Human Heuristic Engine

A lightweight, regex-driven heuristic system for matching contextual signals and emitting advice/predictions with adaptive confidence.

Design goals:

Readable, typed, and PEP8-compliant

Pure-Python, zero external deps

Safe defaults (escaped patterns, clamped confidences)


Public API (essentials):

HeuristicRule: encapsulates one heuristic

HumanHeuristicEngine: manages a list of rules and orchestration


""" from future import annotations

from dataclasses import dataclass, field from datetime import datetime, timezone, timedelta from typing import Any, Dict, List, Optional import re import uuid import random

----------------------------

Utility helpers

----------------------------

def _utcnow_iso() -> str: return datetime.now(tz=timezone.utc).isoformat()

def _parse_iso(dt_str: str) -> datetime: """Best-effort ISO 8601 parser; falls back to naive UTC now. Why: to avoid crashes if stored strings are malformed. """ try: return datetime.fromisoformat(dt_str) except Exception: return datetime.now(tz=timezone.utc)

----------------------------

Heuristic Rule

----------------------------

@dataclass class HeuristicRule: """Represents a single heuristic rule using a regex pattern.

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
times_applied: Number of *successful* matches.
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
    """Attempt to apply the rule to a context.

    Returns a decision dict *only if* the pattern matches. Otherwise None.
    Why: We only count real, successful applications.
    """
    if not context:
        return None

    if re.search(self.pattern, context, flags):
        self.times_applied += 1
        self.last_applied_at = _utcnow_iso()
        decision = {
            "rule_id": self.id,
            "rule_applied": self.name,
            "response": self.response,
            "confidence": self.confidence,
            "tint": self.emotional_tint,
            "rule_strength": self.current_strength(),
            "context_match": True,
        }
        return decision
    return None

def update_feedback(self, was_correct: bool) -> None:
    """Adjust confidence using feedback.
    Why: Keep confidence calibrated with outcomes.
    """
    # Clamp to 0..1 to avoid drift
    def _clamp(x: float) -> float:
        return max(0.0, min(1.0, x))

    if was_correct:
        self.confidence = _clamp(self.confidence + 0.05)
    else:
        self.times_failed += 1
        # Reduce more when failure ratio is high
        reduction = (self.failure_ratio() * 0.5)
        self.confidence = _clamp(max(0.1, self.confidence * (1 - reduction)))

def adjust_tint_intensity(self, resonance: float) -> None:
    """Mutate `emotional_tint` to reflect resonance strength.
    Why: Optional affect adjustment for downstream UX.
    """
    try:
        r = float(resonance)
    except (TypeError, ValueError):
        return
    if r > 0.8:
        self.emotional_tint = f"deep_{self.emotional_tint}"
    elif r < 0.3:
        self.emotional_tint = f"faint_{self.emotional_tint}"

# --- derived metrics ---
def failure_ratio(self) -> float:
    denom = max(1, self.times_applied)
    return self.times_failed / denom

def current_strength(self) -> str:
    denom = max(1, self.times_applied)
    ratio = (self.times_applied - self.times_failed) / denom
    if ratio > 0.9:
        return "trusted instinct"
    if ratio < 0.4:
        return "shaky intuition"
    return "growing awareness"

----------------------------

Engine

----------------------------

class HumanHeuristicEngine: """Orchestrates creation, application, and evolution of heuristic rules."""

def __init__(self) -> None:
    self.heuristics: List[HeuristicRule] = []

# --- creation & inspection ---
def create_heuristic(
    self,
    experience: Dict[str, Any],
    logic_score: float,
    emotion_score: float,
    *,
    use_regex: bool = False,
    regex_flags: int = re.IGNORECASE,
) -> HeuristicRule:
    """Create a rule from an `experience` blob.

    Expected `experience` shape:
    {
      "id": str,
      "input": {"situation": str, ...},
      "reflection": str,
      "emotional_valence": "positive"|"negative"|"neutral"
    }
    """
    input_block = experience.get("input", {}) if isinstance(experience, dict) else {}
    situation = str(input_block.get("situation", "unknown-pattern"))
    origin_id = str(experience.get("id", "unknown"))
    valence = str(experience.get("emotional_valence", "neutral"))
    tint = self._derive_emotional_tint(emotion_score, valence)
    confidence = max(0.0, min(1.0, (float(logic_score) + float(emotion_score)) / 2.0))

    pattern = situation if use_regex else re.escape(situation)

    rule = HeuristicRule(
        name=f"Heuristic-{len(self.heuristics) + 1}",
        pattern=pattern,
        response=str(experience.get("reflection", "")),
        confidence=confidence,
        origin=origin_id,
        emotional_tint=tint,
    )
    self.heuristics.append(rule)
    return rule

def show_rules(self) -> List[Dict[str, Any]]:
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
        }
        for r in self.heuristics
    ]

# --- application & feedback ---
def match_and_apply(self, current_context: str, flags: int = re.IGNORECASE) -> List[Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []
    for rule in self.heuristics:
        decision = rule.apply(current_context, flags=flags)
        if decision:
            decisions.append(decision)
    if decisions:
        return decisions
    return [{"message": "No matching human instincts. Proceed carefully."}]

def revise_heuristic(self, rule_id: str, was_correct: bool) -> bool:
    rule = self._get_rule(rule_id)
    if not rule:
        return False
    rule.update_feedback(was_correct)
    return True

def adjust_rule_tint_intensity(self, rule_id: str, resonance: float) -> bool:
    rule = self._get_rule(rule_id)
    if not rule:
        return False
    rule.adjust_tint_intensity(resonance)
    return True

# --- evolution ---
def synthesize_heuristics(
    self,
    *,
    min_success_ratio: float = 0.85,
    max_new_heuristics: int = 3,
    non_greedy: bool = True,
) -> List[HeuristicRule]:
    """Create new heuristics by combining two highly successful rules."""
    successful_rules: List[HeuristicRule] = [
        r
        for r in self.heuristics
        if (r.times_applied - r.times_failed) / max(1, r.times_applied) > min_success_ratio
        and r.times_applied > 0
    ]

    if len(successful_rules) < 2:
        return []

    new_rules: List[HeuristicRule] = []
    max_iters = min(max_new_heuristics, (len(successful_rules) // 2) + 1)
    for _ in range(max_iters):
        rule_a, rule_b = random.sample(successful_rules, 2)
        middle = ".*?" if non_greedy else ".*"
        new_pattern = f"({rule_a.pattern}){middle}({rule_b.pattern})"
        new_response = f"Synthesis of '{rule_a.response}' and '{rule_b.response}'."
        new_confidence = min(1.0, (rule_a.confidence + rule_b.confidence) / 1.5)

        new_rule = HeuristicRule(
            name=f"Synthesized-{len(self.heuristics) + len(new_rules) + 1}",
            pattern=new_pattern,
            response=new_response,
            confidence=new_confidence,
            origin=f"{rule_a.id}|{rule_b.id}",
            emotional_tint="unified_insight",
        )
        self.heuristics.append(new_rule)
        new_rules.append(new_rule)
    return new_rules

# --- maintenance ---
def decay_unused_rules(self, threshold_days: int = 30, daily_decay: float = 0.95) -> None:
    """Decay confidence for rules unused for `threshold_days`.
    Why: Slowly phase out stale heuristics without abrupt drops.
    """
    now = datetime.now(tz=timezone.utc)
    for r in self.heuristics:
        last_used_str = r.last_applied_at or r.created_at
        last_used_dt = _parse_iso(last_used_str)
        if (now - last_used_dt).days > threshold_days:
            r.confidence *= daily_decay
            if r.confidence < 0.05:
                r.confidence = 0.05

# --- internals ---
def _derive_emotional_tint(self, score: float, valence: str) -> str:
    try:
        s = float(score)
    except (TypeError, ValueError):
        s = 0.0
    v = (valence or "").lower()
    if v == "negative":
        return "regret" if s > 0.7 else "pain"
    if v == "positive":
        return "hope" if s > 0.7 else "naivete"
    return "detachment"

def _get_rule(self, rule_id: str) -> Optional[HeuristicRule]:
    for r in self.heuristics:
        if r.id == rule_id:
            return r
    return None

----------------------------

Example minimal usage (manual test)

----------------------------

if name == "main": engine = HumanHeuristicEngine() exp = { "id": "exp-1", "input": {"situation": "deadline missed"}, "reflection": "Pad estimates by 30% next time.", "emotional_valence": "negative", } rule = engine.create_heuristic(exp, logic_score=0.8, emotion_score=0.6)

print(engine.match_and_apply("Postmortem: we deadline missed due to scope creep."))
engine.revise_heuristic(rule.id, was_correct=False)
print(engine.show_rules())

