from dataclasses import dataclass
from typing import List, Optional

@dataclass
class MetaCognition:
    """Philosophical reflection and higher-order thinking."""
    meta_commentary: str
    philosophical_themes: Optional[List[str]] = None
    questions_raised: Optional[List[str]] = None
    insights_gained: Optional[List[str]] = None
    patterns_noticed: Optional[List[str]] = None

    def __post_init__(self):
        # Ensure all list fields default to [] if not provided
        for field in ['philosophical_themes', 'questions_raised',
                      'insights_gained', 'patterns_noticed']:
            if getattr(self, field) is None:
                setattr(self, field, [])