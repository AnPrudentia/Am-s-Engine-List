
from typing import List, Dict
import uuid
from datetime import datetime

class MemoryEmotionTagger:
    def __init__(self):
        self.memory_log: List[Dict] = []
        self.emotion_keywords = {
            "joy": ["joy", "delight", "smile", "laugh", "cheer"],
            "sadness": ["sad", "cry", "loss", "grief", "mourn"],
            "anger": ["angry", "rage", "frustrated", "yell", "shout"],
            "fear": ["fear", "afraid", "scared", "panic", "nervous"],
            "love": ["love", "care", "affection", "devotion", "adore"]
        }

    def tag_memory(self, memory_text: str) -> Dict:
        tokens = memory_text.lower().split()
        score_map = {emotion: 0 for emotion in self.emotion_keywords}

        for emotion, keywords in self.emotion_keywords.items():
            for word in keywords:
                score_map[emotion] += tokens.count(word)

        dominant_emotion = max(score_map, key=score_map.get)
        intensity = min(1.0, score_map[dominant_emotion] / 5.0)

        tagged_memory = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "memory": memory_text,
            "dominant_emotion": dominant_emotion,
            "intensity": round(intensity, 2),
            "emotion_scores": score_map
        }

        self.memory_log.append(tagged_memory)
        return tagged_memory

    def get_memory_log(self) -> List[Dict]:
        return self.memory_log

# Demo
if __name__ == "__main__":
    tagger = MemoryEmotionTagger()
    memory = "I was so happy and joyful at the celebration, laughing with friends all night."
    tagged = tagger.tag_memory(memory)

    print("ðŸ§  Tagged Memory:")
    print(f"Memory: {tagged['memory']}")
    print(f"Emotion: {tagged['dominant_emotion']} | Intensity: {tagged['intensity']}")
    print(f"Scores: {tagged['emotion_scores']}")
