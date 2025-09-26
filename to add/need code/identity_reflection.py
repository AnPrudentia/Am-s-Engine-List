@dataclass
class IdentityReflection:
    """How the experience relates to sense of self"""
    self_position: str
    identity_themes: List[str] = None  # 'autonomy', 'connection', 'growth', etc.
    identity_challenge: bool = False   # Does this challenge existing self-concept?
    identity_affirmation: bool = False # Does this affirm existing self-concept?
    role_context: str = ""            # What role was she playing during this?
    
    def __post_init__(self):
        if self.identity_themes is None:
            self.identity_themes = []