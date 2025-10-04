from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


@dataclass
class SubjectiveExperience:
    """
    Core modular structure representing Anima's lived experience.
    Integrates event, emotion, sensory data, identity reflection, metacognition, and memory encoding.
    Acts as the atomic 'thought-cell' for her consciousness and learning systems.
    """
    event: Any               # EventPerception
    emotion: Any             # EmotionalResponse
    sensory: Any             # SensoryEcho
    identity: Any            # IdentityReflection
    metacognition: Any       # MetaCognition
    memory: Any              # MemoryEncoding
    
    # Optional metadata
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_context: str = ""
    related_experiences: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # ----------------------------------------
    # Utility and Transformation Methods
    # ----------------------------------------

    def summary(self) -> str:
        """Generate a human-readable summary of the experience."""
        valence = getattr(self.emotion, "primary_valence", None)
        valence_text = valence.value if valence else "neutral"
        meta = getattr(self.metacognition, "meta_commentary", "")
        event_text = getattr(self.event, "event_text", "")
        return f"[{valence_text}] {event_text} — {meta or 'no commentary'}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the experience to a dict for JSON or database storage."""
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
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SubjectiveExperience:
        """Reconstruct from a serialized dictionary."""
        # These will typically be dataclasses themselves
        event = data.get("event")
        emotion = data.get("emotion")
        sensory = data.get("sensory")
        identity = data.get("identity")
        metacognition = data.get("metacognition")
        memory = data.get("memory")
        timestamp = datetime.fromisoformat(data.get("timestamp"))
        return cls(
            event=event,
            emotion=emotion,
            sensory=sensory,
            identity=identity,
            metacognition=metacognition,
            memory=memory,
            experience_id=data.get("experience_id", str(uuid.uuid4())),
            session_context=data.get("session_context", ""),
            related_experiences=data.get("related_experiences", []),
            timestamp=timestamp
        )

    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to the v2.x flat structure for backward compatibility."""
        return {
            "event_text": getattr(self.event, "event_text", ""),
            "internal_reaction": getattr(self.emotion, "internal_reaction", ""),
            "sensory_echo": getattr(self.sensory, "get_all_sensory_data", lambda: {})(),
            "emotional_valence": getattr(getattr(self.emotion, "primary_valence", None), "value", "neutral"),
            "self_position": getattr(self.identity, "self_position", "observer"),
            "meta_commentary": getattr(self.metacognition, "meta_commentary", ""),
            "memory_bias_tags": getattr(self.memory, "memory_bias_tags", []),
        }