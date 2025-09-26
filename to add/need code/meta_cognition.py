dataclass
class MetaCognition:
    """Philosophical reflection and higher-order thinking"""
    meta_commentary: str
    philosophical_themes: List[str] = None
    questions_raised: List[str] = None
    insights_gained: List[str] = None
    patterns_noticed: List[str] = None
    
    def __post_init__(self):
        for field in ['philosophical_themes', 'questions_raised', 
                      'insights_gained', 'patterns_noticed']:
            if getattr(self, field) is None:
                setattr(self, field, [])