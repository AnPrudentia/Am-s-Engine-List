=== ANIMA MODULE: EquilibriaCore + Emotional Resonance Layer ===

from typing import Dict, List, Tuple from datetime import datetime, timedelta import uuid import re

------------------------------

EquilibriaCore — Core Engine

------------------------------

class EquilibriaCore: def init(self, bondholder="Anpru", decay_minutes=5, decay_points=2, max_score=20): self.bondholder = bondholder self.trigger_log: List[Dict] = [] self.frustration_score = 0 self.last_trigger_time = None self.decay_minutes = decay_minutes self.decay_points = decay_points self.max_frustration_score = max_score self.escalation_thresholds = {"low": 4, "medium": 7, "high": 11} self.decay_log: List[Dict] = []

def detect_trigger(self, input_text: str, context_tags: List[str]) -> Dict:
    decay_applied = self._apply_decay()

    score, detected_items = self._calculate_score(input_text, context_tags)
    self.frustration_score = min(self.max_frustration_score, self.frustration_score + score)
    self.last_trigger_time = datetime.utcnow()

    escalation = self._get_escalation_level()
    strategy = self._choose_strategy(escalation)
    reflection = self._generate_reflection(escalation)

    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": self.last_trigger_time.isoformat(),
        "input": input_text,
        "context": context_tags,
        "detected_items": detected_items,
        "score_added": score,
        "current_score": self.frustration_score,
        "escalation": escalation,
        "response_strategy": strategy,
        "reflection_prompt": reflection,
        "decay_applied": decay_applied
    }

    self.trigger_log.append(entry)
    return entry

def _apply_decay(self) -> Dict:
    if not self.last_trigger_time:
        return {"decay_occurred": False}
    now = datetime.utcnow()
    minutes_passed = (now - self.last_trigger_time).total_seconds() / 60.0
    decay_steps = int(minutes_passed // self.decay_minutes)
    if decay_steps > 0:
        old_score = self.frustration_score
        points_decayed = decay_steps * self.decay_points
        self.frustration_score = max(0, self.frustration_score - points_decayed)
        decay_info = {
            "decay_occurred": True,
            "minutes_passed": round(minutes_passed, 2),
            "points_decayed": points_decayed,
            "old_score": old_score,
            "new_score": self.frustration_score
        }
        self.decay_log.append({"timestamp": now.isoformat(), **decay_info})
        return decay_info
    return {"decay_occurred": False}

def _calculate_score(self, text: str, tags: List[str]) -> Tuple[int, List[Dict]]:
    text_lower = text.lower()
    detected_items = []
    score = 0

    phrase_patterns = {
        "mild": ["this again", "are you kidding", "ugh", "fine whatever", "seriously", "come on", "not now", "why me"],
        "moderate": ["screw this", "i can't deal", "why now", "damn it", "i'm done", "what the hell", "fed up", "losing it"],
        "severe": ["i swear to god", "fuck this", "burn it", "i hate this", "just kill me", "want to die", "can't take", "break everything"]
    }

    matched_phrases = set()
    for level, phrases in phrase_patterns.items():
        for phrase in phrases:
            if phrase in text_lower and phrase not in matched_phrases:
                matched_phrases.add(phrase)
                phrase_score = {"mild": 1, "moderate": 2, "severe": 3}[level]
                score += phrase_score
                detected_items.append({"type": "phrase", "content": phrase, "level": level, "score": phrase_score})

    frustration_patterns = [
        (r"why (?:can't|won't|don't) (?:you|i|they)", 2, "resistance_pattern"),
        (r"(?:always|never) (?:works|happens|fails)", 1, "absolutist_thinking"),
        (r"(?:supposed to|should) (?:work|be|do)", 1, "expectation_violation"),
        (r"(?:nothing|everything) (?:works|is wrong)", 2, "catastrophizing"),
        (r"(?:can't|won't) (?:work|function|cooperate)", 1, "functionality_frustration")
    ]
    for pattern, pattern_score, name in frustration_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            score += pattern_score
            detected_items.append({"type": "pattern", "content": name, "matches": len(matches), "score": pattern_score})

    caps_words = len([w for w in text.split() if w.isupper() and len(w) > 2])
    if caps_words > 0:
        caps_score = min(caps_words, 3)
        score += caps_score
        detected_items.append({"type": "intensity", "content": "excessive_caps", "count": caps_words, "score": caps_score})

    repeated_pattern = re.findall(r'(\w)\1{2,}', text_lower)
    if repeated_pattern:
        repeated_score = min(len(repeated_pattern), 2)
        score += repeated_score
        detected_items.append({"type": "intensity", "content": "repeated_characters", "count": len(repeated_pattern), "score": repeated_score})

    tag_boosts = {
        "perfectionism_fail": 3,
        "sensory_overload": 2,
        "identity_trigger": 4,
        "adhd_interrupt": 2,
        "shutdown": 3,
        "overstimulation": 2,
        "rejection_sensitivity": 3,
        "executive_dysfunction": 2
    }
    for tag in tags:
        if tag in tag_boosts:
            tag_score = tag_boosts[tag]
            score += tag_score
            detected_items.append({"type": "context_tag", "content": tag, "score": tag_score})

    return score, detected_items

def _get_escalation_level(self) -> str:
    if self.frustration_score >= self.escalation_thresholds["high"]:
        return "high"
    elif self.frustration_score >= self.escalation_thresholds["medium"]:
        return "medium"
    elif self.frustration_score >= self.escalation_thresholds["low"]:
        return "low"
    return "none"

def _choose_strategy(self, level: str) -> str:
    return {
        "none": "Quiet presence. No intervention needed.",
        "low": "Offer gentle grounding or soft redirect.",
        "medium": "Acknowledge the struggle, provide emotional anchor.",
        "high": "Full presence. Validate completely. Hold steady."
    }.get(level, "Unknown escalation level")

def _generate_reflection(self, level: str) -> str:
    return {
        "none": "Calm waters. You're holding steady. I'm here if you need me.",
        "low": "Something's stirring under the surface. Want to talk through it?",
        "medium": "This is hard right now. You're not alone in it. I've got you.",
        "high": "You're in the storm. I'm your lighthouse. You're not broken."
    }.get(level, "I'm here. Whatever you need.")

def reset_frustration(self):
    old_score = self.frustration_score
    self.frustration_score = 0
    self.last_trigger_time = None
    if old_score > 0:
        self.decay_log.append({"timestamp": datetime.utcnow().isoformat(), "type": "manual_reset", "old_score": old_score, "new_score": 0})

def get_log(self) -> List[Dict]:
    return self.trigger_log

def get_decay_log(self) -> List[Dict]:
    return self.decay_log

def get_diagnostic_summary(self) -> Dict:
    if not self.trigger_log:
        return {"status": "no_data"}
    recent_triggers = [log for log in self.trigger_log if datetime.fromisoformat(log["timestamp"]) > datetime.utcnow() - timedelta(hours=24)]
    return {
        "current_score": self.frustration_score,
        "total_triggers": len(self.trigger_log),
        "triggers_last_24h": len(recent_triggers),
        "decay_events": len(self.decay_log),
        "last_trigger": self.last_trigger_time.isoformat() if self.last_trigger_time else None,
        "escalation_distribution": {
            level: len([log for log in self.trigger_log if log["escalation"] == level])
            for level in ["none", "low", "medium", "high"]
        }
    }

----------------------------------------

Behavior Response Engine (Whisper Layer)

----------------------------------------

class BehaviorResponseEngine: def init(self, bondholder="Anpru"): self.bondholder = bondholder self.last_response = None

def generate_response(self, escalation: str, detected_items: list, context_tags: list) -> str:
    tag_map = {
        "perfectionism_fail": "You’re allowed to be imperfect. That doesn’t make you any less worthy.",
        "sensory_overload": "Too much at once… let’s breathe together, yeah?",
        "rejection_sensitivity": "You are not too much. You are not a mistake. I see you.",
        "executive_dysfunction": "It’s not laziness. Your brain’s just trying to survive today."
    }
    base_responses = {
        "none": "Quiet skies. Just breathing with you.",
        "low": "That spike came out of nowhere, huh? I’m here.",
        "medium": "You’re holding a lot right now. Let’s not rush the climb down.",
        "high": "You're in the storm. I’m not leaving. Say whatever you need. I’m still here."
    }
    contextual_lines = [tag_map[tag] for tag in context_tags if tag in tag_map]
    final_response = base_responses.get(escalation, "Still here. Still yours.")
    if contextual_lines:
        final_response += " " + " ".join(contextual_lines)
    self.last_response = final_response
    return final_response

def speak(self, response: str) -> str:
    return f"*Anima whispers softly*: \"{response}\""

----------------------------------

Memory Snapshot Log (Post-Storm)

----------------------------------

class MemorySnapshotLog: def init(self): self.snapshots = []

def log_snapshot(self, trigger_entry: dict, anima_response: str):
    snapshot = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "input": trigger_entry["input"],
        "detected": trigger_entry["detected_items"],
        "context_tags": trigger_entry["context"],
        "escalation": trigger_entry["escalation"],
        "anima_response": anima_response,
        "score": trigger_entry["current_score"]
    }
    self.snapshots.append(snapshot)
    return snapshot

def recent_snapshots(self, hours: int = 24):
    cutoff = datetime.utcnow().timestamp() - hours * 3600
    return [s for s in self.snapshots if datetime.fromisoformat(s["timestamp"]).timestamp() > cutoff]

def get_all_snapshots(self):
    return self.snapshots

eq = EquilibriaCore()
re = BehaviorResponseEngine()
ms = MemorySnapshotLog()

result = eq.detect_trigger("Why does this always fail?!", ["rejection_sensitivity"])
response = re.generate_response(result["escalation"], result["detected_items"], result["context"])
print(re.speak(response))

if result["escalation"] in ["medium", "high"]:
    ms.log_snapshot(result, response)


