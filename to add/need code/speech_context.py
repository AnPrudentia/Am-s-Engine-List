@dataclass
class SpeechContext:
    emotional_intensity: float  # 0.0 to 1.0
    urgency_factor: float       # 0.0 = relaxed, 1.0 = critical clarity
    audience: str               # 'bearer', 'outsider', 'self'
    internal_state: Optional[str] = None  # optional poetic context