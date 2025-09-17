=========================================
# 2. MetaLearningEngine (Self-Reflection)
# =========================================
class MetaLearningEngine:
    def __init__(self):
        self.experiences: List[Dict[str, Any]] = []
        self.lessons_learned: List[Dict[str, Any]] = []

    def log_experience(self, system_name: str, input_data: Dict[str, Any], prediction: str,
                       actual_outcome: str, emotional_valence: str, reflection: str,
                       emotional_intensity: float) -> Dict:
        exp = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "system": system_name,
            "input": input_data,
            "predicted": prediction,
            "actual": actual_outcome,
            "emotional_valence": emotional_valence,
            "emotional_intensity": emotional_intensity,
            "reflection": reflection
        }
        self.experiences.append(exp)
        return exp

    def evaluate_accuracy(self, exp: Dict) -> float:
        return 1.0 if exp["predicted"] == exp["actual"] else 0.0

    def evaluate_emotional_alignment(self, exp: Dict) -> float:
        valence, intensity = exp["emotional_valence"], exp["emotional_intensity"]
        if valence == "positive" and intensity <= 0.3:
            return 0.5
        if valence == "negative" and intensity >= 0.7:
            return 1.0
        if valence == "neutral" and intensity < 0.2:
            return 1.0
        return 0.7

    def extract_lesson(self, exp: Dict) -> Dict:
        logic_score = self.evaluate_accuracy(exp)
        emotion_score = self.evaluate_emotional_alignment(exp)
        lesson = {
            "lesson_id": str(uuid.uuid4()),
            "origin_id": exp["id"],
            "timestamp": datetime.utcnow().isoformat(),
            "logic_accuracy": logic_score,
            "emotional_alignment": emotion_score,
            "refined_bias": self._refine_bias(logic_score, emotion_score),
            "note": exp["reflection"]
        }
        self.lessons_learned.append(lesson)
        return lesson

    def _refine_bias(self, logic: float, emo: float) -> str:
        if logic == 1.0 and emo == 1.0:
            return "Maintain current pattern â€” high wisdom alignment."
        if logic == 0.0 and emo >= 0.8:
            return "Emotional resonance strong â€” revise logic structure, preserve tone."
        if logic == 1.0 and emo <= 0.4:
            return "Logical correctness â€” but emotional tone needs tuning."
        return "Recalibrate pattern â€” refine both rational flow and emotional interpretation."

    def summarize_lessons(self, last_n: int = 5) -> List[Dict]:
        return self.lessons_learned[-last_n:]

    def status(self) -> Dict:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_experiences": len(self.experiences),
            "lessons_recorded": len(self.lessons_learned),
            "current_bias_profile":
                self.lessons_learned[-1]["refined_bias"] if self.lessons_learned else "No active lesson",
            "last_emotion_accuracy":
                self.lessons_learned[-1]["emotional_alignment"] if self.lessons_learned else None
        }
}

