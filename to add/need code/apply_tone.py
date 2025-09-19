def _apply_tone(self, message: str, intent: str = "neutral", severity: float = 0.5) -> str:
    """
    Adds a contextual emotional prefix or tone cue to a message.
    
    Parameters:
    - message: The main evaluation content.
    - intent: One of ["neutral", "warning", "gentle", "humorous", "serious", "critical"]
    - severity: A float from 0.0 to 1.0 indicating intensity.

    Returns:
    - A message prefixed with a tone-setting phrase.
    """
    intent = intent.lower()
    tone_prefix = ""

    if intent == "neutral":
        tone_prefix = "Observation:" if severity > 0.6 else ""
    elif intent == "warning":
        tone_prefix = "⚠️ Warning:" if severity > 0.7 else "Note:"
    elif intent == "gentle":
        tone_prefix = "Hmm… maybe consider this:"
    elif intent == "humorous":
        tone_prefix = "😅 Just saying—but:"
    elif intent == "serious":
        tone_prefix = "Important:"
    elif intent == "critical":
        tone_prefix = "🛑 Critical Insight:"
    else:
        tone_prefix = "Insight:"

    return f"{tone_prefix} {message}" if tone_prefix else message
