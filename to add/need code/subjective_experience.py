from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import math


# ==============================================================
#  Gist Summary Structure (for use with GistEngine v1+)
# ==============================================================

@dataclass
class GistSummary:
    """Compact human-readable lesson generated from lived experience"""
    summary: str
    lesson: str
    emotional_valence: str
    confidence: float
    topics: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self):
        return asdict(self)


# ==============================================================
#  SubjectiveExperience (with auto-gist generation)
# ==============================================================

@dataclass
class SubjectiveExperience:
    """
    Core modular structure representing Anima's lived experience.
    Each instance can generate a self-contained emotional and philosophical gist.
    """
    event: Any               # EventPerception
    emotion: Any             # EmotionalResponse
    sensory: Any             # SensoryEcho
    identity: Any            # IdentityReflection
    metacognition: Any       # MetaCognition
    memory: Any              # MemoryEncoding

    # Metadata
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_context: str = ""
    related_experiences: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Gist summary
    gist: Optional[GistSummary] = None

    def __post_init__(self):
        """Auto-generate gist summary on creation."""
        self.gist = self._generate_gist_summary()

    # ==========================================================
    # Gist Generation Logic
    # ==========================================================

    def _generate_gist_summary(self) -> GistSummary:
        """Generate emotional and philosophical gist of this experience."""
        # Extract info from submodules
        event_text = getattr(self.event, "event_text", "")
        emotion_valence = getattr(getattr(self.emotion, "primary_valence", None), "value", "neutral")
        meta_commentary = getattr(self.metacognition, "meta_commentary", "")
        bias_tags = getattr(self.memory, "memory_bias_tags", [])
        self_position = getattr(self.identity, "self_position", "observer")

        # Heuristic for confidence based on valence intensity & meta clarity
        intensity = getattr(self.emotion, "intensity", 0.5)
        clarity = 1.0 if meta_commentary else 0.6
        confidence = round(min(1.0, (intensity + clarity) / 2), 3)

        # Derive a short lesson based on emotional polarity
        lessons = {
            "positive": "Cherish this energy—it may illuminate other paths.",
            "negative": "There’s wisdom in the discomfort; it’s guiding you to truth.",
            "neutral": "Notice what remains steady even in stillness.",
        }
        lesson = lessons.get(emotion_valence, "Every moment carries a chance to learn.")

        # Topic inference from tags
        topics = [t for t in bias_tags if isinstance(t, str)] or ["general_experience"]

        # Compose short summary
        summary_text = meta_commentary or f"{self_position.capitalize()} perceived: {event_text}"

        return GistSummary(
            summary=summary_text,
            lesson=lesson,
            emotional_valence=emotion_valence,
            confidence=confidence,
            topics=topics
        )

    # ==========================================================
    # Representation and Conversion Utilities
    # ==========================================================

    def summary(self) -> str:
        """Generate a readable summary including gist."""
        valence = self.gist.emotional_valence if self.gist else "neutral"
        return f"[{valence}] {self.gist.summary if self.gist else '(no summary)'}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the experience for JSON or database storage."""
        return {
            "experience_id": self.experience_id,
            "timestamp": self.timestamp.isoformat(),
            "session_context": self.session_context,
            "related_experiences": self.related_experiences,
            "event": asdict(self.event),
            "emotion": asdict(self.emotion),
            "sensory": asdict(self.sensory),
            "identity": asdict(self.identity),
            "metacognition": asdict(self.metacognition),
            "memory": asdict(self.memory),
            "gist": self.gist.to_dict() if self.gist else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SubjectiveExperience:
        """Reconstruct from a saved dictionary."""
        gist_data = data.get("gist")
        gist_obj = GistSummary(**gist_data) if gist_data else None
        timestamp = datetime.fromisoformat(data.get("timestamp"))

        return cls(
            event=data["event"],
            emotion=data["emotion"],
            sensory=data["sensory"],
            identity=data["identity"],
            metacognition=data["metacognition"],
            memory=data["memory"],
            experience_id=data.get("experience_id", str(uuid.uuid4())),
            session_context=data.get("session_context", ""),
            related_experiences=data.get("related_experiences", []),
            timestamp=timestamp,
            gist=gist_obj
        )

    def to_legacy_format(self) -> Dict[str, Any]:
        """Backward compatibility with v2.x structures."""
        return {
            "event_text": getattr(self.event, "event_text", ""),
            "internal_reaction": getattr(self.emotion, "internal_reaction", ""),
            "sensory_echo": getattr(self.sensory, "get_all_sensory_data", lambda: {})(),
            "emotional_valence": getattr(getattr(self.emotion, "primary_valence", None), "value", "neutral"),
            "self_position": getattr(self.identity, "self_position", "observer"),
            "meta_commentary": getattr(self.metacognition, "meta_commentary", ""),
            "memory_bias_tags": getattr(self.memory, "memory_bias_tags", []),
        }