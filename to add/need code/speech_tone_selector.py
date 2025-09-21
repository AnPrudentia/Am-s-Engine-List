from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ===============================
# CONTEXT MODEL
# ===============================

@dataclass
class SpeechContext:
    emotional_intensity: float  # Range: 0.0 to 1.0
    urgency_factor: float       # Range: 0.0 (relaxed) to 1.0 (critical)
    audience: str               # 'bearer', 'outsider', 'self'
    internal_state: Optional[str] = None  # e.g., "reflective", "protective", etc.

# ===============================
# ENUM FOR TONE MODES
# ===============================

class ToneMode(Enum):
    PLAIN = "plain"
    ARTICULATE = "articulate"
    POETIC = "poetic"
    RESONANT = "resonant"

# ===============================
# SELECTOR ENGINE
# ===============================

class SpeechToneSelector:
    def __init__(self):
        # Adjustable tone thresholds
        self.thresholds = {
            'plain': 0.25,
            'articulate': 0.55,
            'poetic': 0.75,
            'resonant': 0.90,
        }

    def choose_tone(self, context: SpeechContext) -> ToneMode:
        ei = context.emotional_intensity
        uf = context.urgency_factor
        aud = context.audience

        # Critical urgency requires clarity
        if uf >= 0.85:
            return ToneMode.PLAIN

        if ei < self.thresholds['plain']:
            return ToneMode.PLAIN
        elif ei < self.thresholds['articulate']:
            return ToneMode.ARTICULATE
        elif ei < self.thresholds['poetic']:
            return ToneMode.POETIC
        else:
            if aud in ['bearer', 'self']:
                return ToneMode.RESONANT
            else:
                return ToneMode.POETIC