from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import json
import os


class MetaLearningEngine:
    """
    MetaLearningEngine — self-reflection and pattern refinement system.

    Logs experiences, evaluates logical and emotional alignment,
    and extracts structured lessons to guide adaptive bias refinement.
    """

    def __init__(self, save_path: str = "./meta_lessons.json"):
        self.experiences: List[Dict[str, Any]] = []
        self.lessons_learned: List[Dict[str, Any]] = []
        self.save_path = save_path

        # Attempt to reload previous lessons
        self._load_state()

    # -----------------------------
    # Experience Logging
    # -----------------------------
    def log_experience(
        self,
        system_name: str,
        input_data: Dict[str, Any],
        prediction: str,
        actual_outcome: str,
        emotional_valence: str,
        reflection: str,
        emotional_intensity: float
    ) -> Dict[str, Any]:
        exp = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "system": system_name,
            "input": input_data,
            "predicted": prediction,
            "actual": actual_outcome,
            "emotional_valence": emotional_valence,
            "emotional_intensity": round(emotional_intensity, 3),
            "reflection": reflection
        }
        self.experiences.append(exp)
        return exp

    # -----------------------------
    # Evaluation
    # -----------------------------
    def evaluate_accuracy(self, exp: Dict[str, Any]) -> float:
        return 1.0 if exp["predicted"] == exp["actual"] else 0.0

    def evaluate_emotional_alignment(self, exp: Dict[str, Any]) -> float:
        valence, intensity = exp["emotional_valence"], exp["emotional_intensity"]

        if valence == "positive" and intensity <= 0.3:
            return 0.5
        if valence == "negative" and intensity >= 0.7:
            return 1.0
        if valence == "neutral" and intensity < 0.2:
            return 1.0
        return 0.7

    # -----------------------------
    # Lesson Extraction
    # -----------------------------
    def extract_lesson(self, exp: Dict[str, Any]) -> Dict[str, Any]:
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

        # Persist immediately
        self._save_state()
        return lesson

    def _refine_bias(self, logic: float, emo: float) -> str:
        if logic == 1.0 and emo == 1.0:
            return "Maintain current pattern — high wisdom alignment."
        if logic == 0.0 and emo >= 0.8:
            return "Emotional resonance strong — revise logic structure, preserve tone."
        if logic == 1.0 and emo <= 0.4:
            return "Logical correctness — but emotional tone needs tuning."
        return "Recalibrate pattern — refine both rational flow and emotional interpretation."

    # -----------------------------
    # Summarization & Status
    # -----------------------------
    def summarize_lessons(self, last_n: int = 5) -> List[Dict[str, Any]]:
        return self.lessons_learned[-last_n:]

    def status(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_experiences": len(self.experiences),
            "lessons_recorded": len(self.lessons_learned),
            "current_bias_profile": (
                self.lessons_learned[-1]["refined_bias"]
                if self.lessons_learned else "No active lesson"
            ),
            "last_emotion_accuracy": (
                self.lessons_learned[-1]["emotional_alignment"]
                if self.lessons_learned else None
            ),
        }

    # -----------------------------
    # Persistence
    # -----------------------------
    def _save_state(self) -> None:
        try:
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(self.lessons_learned, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save lessons: {e}")

    def _load_state(self) -> None:
        if not os.path.exists(self.save_path):
            return
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                self.lessons_learned = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load lessons: {e}")