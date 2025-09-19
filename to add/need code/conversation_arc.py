class ConversationArc:
    def __init__(self):
        self.turns = deque(maxlen=20)
        self.emotional_trajectory = deque(maxlen=10)
        self.themes = defaultdict(int)
        self.depth_progression = 0.0
    def add_turn(self, user_input: str, emotion_context: Dict[str, float]):
        turn = {
            "timestamp": _utcnow(), "input": user_input, "emotions": emotion_context,
            "word_count": len(user_input.split()), "depth_indicators": self._assess_depth(user_input)
        }
        self.turns.append(turn); self.emotional_trajectory.append(emotion_context)
        self._update_themes(user_input); self._update_depth_progression()
    def current_state(self) -> str:
        if len(self.turns) < 2: return "opening"
        recent = list(self.emotional_trajectory)[-3:]
        crisis = {"sadness","fear","anger","overwhelm","despair"}
        cnt = sum(1 for ems in recent for e in ems.keys() if e in crisis)
        if cnt >= 2: return "crisis"
        if self.depth_progression > 0.7: return "deep"
        if len(self.turns) > 5: return "flowing"
        return "warming"
    def depth_level(self) -> float: return self.depth_progression
    def _assess_depth(self, ui: str) -> float:
        depth_words = {"meaning","purpose","why","soul","deep","profound","existential","philosophy","truth","essence","core"}
        words = ui.lower().split(); score = sum(1 for w in words if w in depth_words)
        length = min(0.3, len(words)/100); q = 0.2 if "?" in ui else 0
        return min(1.0, score * 0.1 + length + q)
    def _update_themes(self, ui: str):
        theme_words = {"relationship","work","family","love","growth","struggle","future","past","healing","change","fear","hope"}
        for w in ui.lower().split():
            if w in theme_words: self.themes[w] += 1
    def _update_depth_progression(self):
        if not self.turns: return
        recent = sum(t["depth_indicators"] for t in list(self.turns)[-3:])
        avg = recent / min(3, len(self.turns))
        self.depth_progression = 0.7 * self.depth_progression + 0.3 * avg
