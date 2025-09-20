class EmotionalGravity:
    def __init__(self, state: str = "Present pain, unseen"):
        self.state = state

    def pull(self, signal: str) -> str:
        gravity_responses = [
            f"{signal} (The weight is acknowledged.)",
            f"{signal} (Held with the gravity of {self.state.lower()}.)",
            f"{signal} (This heaviness has meaning.)",
            f"{signal} (The depth is witnessed.)"
        ]
        return random.choice(gravity_responses)