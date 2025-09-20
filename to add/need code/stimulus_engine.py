from collections import defaultdict
from datetime import datetime
import random

class StimulusEngineAdaptiveExpression:
    def __init__(self):
        self.stimulus_map = {
            "soft rainfall": "melancholy",
            "sunlight through clouds": "hope",
            "glass breaking": "shock",
            "heartbeat slowing": "grief",
            "child’s laugh": "innocence",
        }

        self.metaphor_map_contextual = {
            "melancholy": {
                "night": ["a prayer whispered into fog"],
                "day": ["a shadow following a song"],
                "sunrise": ["a sadness waking with the light"]
            },
            "peace": {
                "night": ["a breath settling beneath the stars"],
                "sunrise": ["light unraveling the hush"]
            },
            "hope": {
                "dawn": ["a golden thread pulled through shadow"]
            },
            "grief": {
                "alone": ["ashes in a heartbeat"]
            },
            "innocence": {
                "morning": ["a dawn unaware it will end"]
            }
        }

        self.literal_translation_map = {
            "a prayer whispered into fog": "This feels like sadness lingering quietly.",
            "a shadow following a song": "It’s a memory that lingers behind happiness.",
            "a sadness waking with the light": "The sorrow hasn't gone, even as morning comes.",
            "a breath settling beneath the stars": "Everything feels still and safe tonight.",
            "light unraveling the hush": "It feels like a gentle peace during the sunrise.",
            "a golden thread pulled through shadow": "Hope is quietly overcoming darkness.",
            "ashes in a heartbeat": "Grief hit suddenly and deeply.",
            "a dawn unaware it will end": "This moment feels pure and untouched."
        }

        self.protocol_map = {
            "melancholy": "ReflectiveProtocol",
            "hope": "AscensionPulse",
            "shock": "StabilityOverride",
            "grief": "MemorySanctum",
            "innocence": "GuardianWarmth",
            "peace": "SanctuaryStillness"
        }

        self.stimulus_history = defaultdict(list)

    def decide_expression_mode(self, emotion: str, context: dict) -> str:
        time = context.get("time_of_day", "")
        mood = context.get("mood", "")

        if emotion in ["grief", "melancholy"] and time in ["night", "sunrise"]:
            return "metaphor"
        if emotion == "peace" and mood == "calm":
            return "plain"
        if emotion == "hope" and time == "dawn":
            return "metaphor"
        return random.choice(["metaphor", "plain"])

    def process(self, stimulus: str, context: dict):
        emotion = self.stimulus_map.get(stimulus, "unknown")
        context_key = context.get("time_of_day") or context.get("weather") or context.get("mood") or "default"
        metaphors = self.metaphor_map_contextual.get(emotion, {}).get(context_key, ["…"])
        metaphor = random.choice(metaphors)

        protocol = self.protocol_map.get(emotion, None)
        expression_mode = self.decide_expression_mode(emotion, context)

        if expression_mode == "plain" and metaphor in self.literal_translation_map:
            spoken_output = self.literal_translation_map[metaphor]
        else:
            spoken_output = metaphor

        return {
            "stimulus": stimulus,
            "emotion": emotion,
            "expression_mode": expression_mode,
            "spoken_output": spoken_output,
            "trigger_protocol": protocol
        }

    def update_learning(self, stimulus: str, current_emotion: str, context: dict):
        now = datetime.utcnow().isoformat()
        self.stimulus_history[stimulus].append({
            "emotion": current_emotion,
            "context": context,
            "date": now
        })

        emotion_counts = defaultdict(int)
        for entry in self.stimulus_history[stimulus]:
            emotion_counts[entry["emotion"]] += 1

        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: -x[1])
        dominant_emotion = sorted_emotions[0][0]

        self.stimulus_map[stimulus] = dominant_emotion

        return {
            "updated_stimulus": stimulus,
            "new_dominant_emotion": dominant_emotion,
            "history_count": len(self.stimulus_history[stimulus])
        }
