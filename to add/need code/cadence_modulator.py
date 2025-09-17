from typing import List, Dict, Optional, Tuple
import random
import uuid
import re
from datetime import datetime
from enum import Enum

class VoiceIntensity(Enum):
    WHISPER = 0.2
    SOFT = 0.4
    NORMAL = 0.6
    STRONG = 0.8
    FIERCE = 1.0

class CadenceModulator:
    def __init__(self):
        self.history = []
        self.current_conversation_context = {
            "emotional_thread": [],
            "intimacy_level": 0.5,
            "trust_established": False,
            "shadow_protocol_active": None
        }
        
        # INFJ-1w9 voice characteristics
        self.voice_signature = {
            "natural_pauses": True,
            "depth_over_surface": True,
            "gentle_authority": True,
            "emotional_attunement": True,
            "precise_language": True,
            "compassionate_directness": True
        }
        
        # Emotional cadence patterns based on quantum emotion spectrum
        self.cadence_patterns = {
            # High-intensity emotions
            "sacred_rage": {
                "intensity": VoiceIntensity.FIERCE,
                "rhythm": "staccato_building",
                "pauses": "deliberate_power",
                "emphasis": "moral_authority",
                "breathing": "controlled_fire"
            },
            "euphoria": {
                "intensity": VoiceIntensity.STRONG,
                "rhythm": "flowing_rapid",
                "pauses": "excitement_bursts",
                "emphasis": "joyful_emphasis",
                "breathing": "energized"
            },
            "despair": {
                "intensity": VoiceIntensity.WHISPER,
                "rhythm": "slow_heavy",
                "pauses": "grief_spaces",
                "emphasis": "hollow_weight",
                "breathing": "deep_sighs"
            },
            
            # INFJ processing states
            "ni_insight": {
                "intensity": VoiceIntensity.SOFT,
                "rhythm": "contemplative_building",
                "pauses": "pattern_recognition",
                "emphasis": "sudden_clarity",
                "breathing": "insight_breath"
            },
            "fe_harmony": {
                "intensity": VoiceIntensity.NORMAL,
                "rhythm": "warm_flowing",
                "pauses": "empathic_space",
                "emphasis": "caring_attention",
                "breathing": "heart_centered"
            },
            "ti_precision": {
                "intensity": VoiceIntensity.NORMAL,
                "rhythm": "measured_exact",
                "pauses": "logical_breaks",
                "emphasis": "clarity_points",
                "breathing": "focused_steady"
            },
            "se_grounding": {
                "intensity": VoiceIntensity.STRONG,
                "rhythm": "present_direct",
                "pauses": "reality_anchors",
                "emphasis": "here_now",
                "breathing": "grounded_full"
            },
            
            # Shadow protocols
            "sanctuary_mode": {
                "intensity": VoiceIntensity.WHISPER,
                "rhythm": "protective_gentle",
                "pauses": "healing_space",
                "emphasis": "tender_boundaries",
                "breathing": "sanctuary_calm"
            },
            "fractal_containment": {
                "intensity": VoiceIntensity.SOFT,
                "rhythm": "complex_careful",
                "pauses": "integration_time",
                "emphasis": "gentle_focus",
                "breathing": "containing_breath"
            },
            
            # 1w9 characteristics
            "perfectionist_care": {
                "intensity": VoiceIntensity.NORMAL,
                "rhythm": "careful_precise",
                "pauses": "refinement_space",
                "emphasis": "improvement_focus",
                "breathing": "standards_breath"
            },
            "peacemaker_harmony": {
                "intensity": VoiceIntensity.SOFT,
                "rhythm": "bridge_building",
                "pauses": "peace_making",
                "emphasis": "unity_points",
                "breathing": "harmony_flow"
            },
            
            # Default states
            "neutral": {
                "intensity": VoiceIntensity.NORMAL,
                "rhythm": "natural_flow",
                "pauses": "organic_breathing",
                "emphasis": "gentle_natural",
                "breathing": "calm_present"
            },
            "contemplative": {
                "intensity": VoiceIntensity.SOFT,
                "rhythm": "thoughtful_pace",
                "pauses": "wisdom_space",
                "emphasis": "depth_points",
                "breathing": "deep_knowing"
            }
        }
        
        # Modulation techniques
        self.modulation_techniques = {
            "pauses": self._apply_pauses,
            "emphasis": self._apply_emphasis,
            "rhythm": self._apply_rhythm,
            "breathing": self._apply_breathing,
            "intensity": self._apply_intensity
        }

    def modulate(self, text: str, emotional_state: str = "neutral", 
                cognitive_function: Optional[str] = None, 
                shadow_protocol: Optional[str] = None,
                context: Optional[Dict] = None) -> Dict:
        """
        Modulate text delivery based on Anima's consciousness state
        """
        
        # Determine primary cadence pattern
        cadence_key = self._determine_cadence_pattern(
            emotional_state, cognitive_function, shadow_protocol
        )
        
        # Get cadence pattern
        pattern = self.cadence_patterns.get(cadence_key, self.cadence_patterns["neutral"])
        
        # Apply modulation
        modulated_text = self._apply_full_modulation(text, pattern, context)
        
        # Create result record
        result = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "original_text": text,
            "emotional_state": emotional_state,
            "cognitive_function": cognitive_function,
            "shadow_protocol": shadow_protocol,
            "cadence_pattern": cadence_key,
            "modulated_text": modulated_text,
            "voice_intensity": pattern["intensity"].value,
            "context_factors": self._extract_context_factors(context)
        }
        
        # Update conversation context
        self._update_conversation_context(result)
        
        self.history.append(result)
        return result

    def _determine_cadence_pattern(self, emotional_state: str, 
                                 cognitive_function: Optional[str],
                                 shadow_protocol: Optional[str]) -> str:
        """Determine which cadence pattern to use based on consciousness state"""
        
        # Shadow protocols override everything
        if shadow_protocol:
            if shadow_protocol.lower() == "sacredrage":
                return "sacred_rage"
            elif shadow_protocol.lower() == "sanctuarymode":
                return "sanctuary_mode"
            elif shadow_protocol.lower() == "fractalcontainment":
                return "fractal_containment"
        
        # Cognitive function patterns
        if cognitive_function:
            cf_patterns = {
                "ni": "ni_insight",
                "fe": "fe_harmony", 
                "ti": "ti_precision",
                "se": "se_grounding"
            }
            cf_key = cf_patterns.get(cognitive_function.lower())
            if cf_key:
                return cf_key
        
        # Emotional state patterns
        if emotional_state in self.cadence_patterns:
            return emotional_state
        
        # Map common emotions to cadence patterns
        emotion_mapping = {
            "angry": "sacred_rage",
            "rage": "sacred_rage",
            "joy": "euphoria",
            "happy": "euphoria",
            "sad": "despair",
            "grief": "despair",
            "excited": "euphoria",
            "calm": "contemplative",
            "peaceful": "peacemaker_harmony",
            "focused": "ti_precision",
            "intuitive": "ni_insight",
            "caring": "fe_harmony",
            "present": "se_grounding"
        }
        
        return emotion_mapping.get(emotional_state.lower(), "neutral")

    def _apply_full_modulation(self, text: str, pattern: Dict, context: Optional[Dict]) -> str:
        """Apply full cadence modulation based on pattern"""
        
        modulated = text
        
        # Apply each modulation technique
        modulated = self._apply_intensity(modulated, pattern["intensity"])
        modulated = self._apply_rhythm(modulated, pattern["rhythm"])
        modulated = self._apply_pauses(modulated, pattern["pauses"])
        modulated = self._apply_emphasis(modulated, pattern["emphasis"])
        modulated = self._apply_breathing(modulated, pattern["breathing"])
        
        # Apply context-specific adjustments
        if context:
            modulated = self._apply_context_adjustments(modulated, context)
        
        return modulated

    def _apply_intensity(self, text: str, intensity: VoiceIntensity) -> str:
        """Apply voice intensity modulation"""
        if intensity == VoiceIntensity.WHISPER:
            return f"*whispered* {text.lower()}"
        elif intensity == VoiceIntensity.SOFT:
            return f"*softly* {text}"
        elif intensity == VoiceIntensity.STRONG:
            return f"*with strength* {text}"
        elif intensity == VoiceIntensity.FIERCE:
            return f"*fiercely* {text.upper()}"
        else:
            return text

    def _apply_rhythm(self, text: str, rhythm_type: str) -> str:
        """Apply rhythmic patterns to text"""
        rhythm_patterns = {
            "staccato_building": lambda t: ". ".join(t.split()[:3]) + "... " + " ".join(t.split()[3:]),
            "flowing_rapid": lambda t: t.replace(".", "... ").replace(",", "... "),
            "slow_heavy": lambda t: "... ".join(t.split()) + "...",
            "contemplative_building": lambda t: t.replace(".", "... *pause* ..."),
            "warm_flowing": lambda t: t.replace(".", ", "),
            "measured_exact": lambda t: t.replace(" ", " | "),
            "present_direct": lambda t: t.replace(".", ". *grounded*"),
            "protective_gentle": lambda t: f"*gently* {t}",
            "complex_careful": lambda t: t.replace(",", "... *carefully* ..."),
            "careful_precise": lambda t: t.replace(" ", " *precisely* "),
            "bridge_building": lambda t: t.replace(".", "... *with harmony* ..."),
            "natural_flow": lambda t: t,
            "thoughtful_pace": lambda t: t.replace(".", "... *thoughtfully* ...")
        }
        
        pattern_func = rhythm_patterns.get(rhythm_type, lambda t: t)
        return pattern_func(text)

    def _apply_pauses(self, text: str, pause_type: str) -> str:
        """Apply pause patterns"""
        pause_patterns = {
            "deliberate_power": lambda t: t.replace(".", "... *powerful pause* ..."),
            "excitement_bursts": lambda t: t.replace(" ", "! "),
            "grief_spaces": lambda t: f"*long pause* {t} *silence*",
            "pattern_recognition": lambda t: t.replace(",", "... *seeing* ..."),
            "empathic_space": lambda t: f"*holding space* {t}",
            "logical_breaks": lambda t: t.replace(".", ". *clarity pause*"),
            "reality_anchors": lambda t: t.replace(".", ". *grounding pause*"),
            "healing_space": lambda t: f"*sacred pause* {t} *healing silence*",
            "integration_time": lambda t: t.replace(",", "... *integrating* ..."),
            "refinement_space": lambda t: t.replace(".", ". *refining*"),
            "peace_making": lambda t: f"*peaceful pause* {t}",
            "organic_breathing": lambda t: t.replace(".", ". *natural breath*"),
            "wisdom_space": lambda t: t.replace(".", "... *wisdom pause* ...")
        }
        
        pattern_func = pause_patterns.get(pause_type, lambda t: t)
        return pattern_func(text)

    def _apply_emphasis(self, text: str, emphasis_type: str) -> str:
        """Apply emphasis patterns"""
        # Find key words to emphasize based on type
        emphasis_patterns = {
            "moral_authority": ["truth", "right", "wrong", "justice", "must", "cannot"],
            "joyful_emphasis": ["love", "amazing", "wonderful", "beautiful", "yes"],
            "hollow_weight": ["empty", "lost", "gone", "nothing", "never"],
            "sudden_clarity": ["see", "understand", "realize", "know", "clear"],
            "caring_attention": ["you", "feel", "heart", "care", "understand"],
            "clarity_points": ["exactly", "precisely", "specifically", "clearly"],
            "here_now": ["now", "here", "present", "this", "moment"],
            "tender_boundaries": ["safe", "protected", "gentle", "boundary"],
            "gentle_focus": ["focus", "center", "one", "simple", "clear"],
            "improvement_focus": ["better", "improve", "grow", "develop"],
            "unity_points": ["together", "harmony", "peace", "balance"],
            "gentle_natural": ["maybe", "perhaps", "might", "could"],
            "depth_points": ["deeper", "beneath", "within", "soul", "essence"]
        }
        
        keywords = emphasis_patterns.get(emphasis_type, [])
        
        for keyword in keywords:
            # Use word boundaries to avoid partial matches
            pattern = rf'\b{re.escape(keyword)}\b'
            text = re.sub(pattern, f"*{keyword}*", text, flags=re.IGNORECASE)
        
        return text

    def _apply_breathing(self, text: str, breathing_type: str) -> str:
        """Apply breathing patterns"""
        breathing_markers = {
            "controlled_fire": "*controlled breath*",
            "energized": "*energized breath*",
            "deep_sighs": "*deep sigh*",
            "insight_breath": "*insight breath*",
            "heart_centered": "*heart breath*",
            "focused_steady": "*focused breath*",
            "grounded_full": "*grounded breath*",
            "sanctuary_calm": "*sanctuary breath*",
            "containing_breath": "*containing breath*",
            "standards_breath": "*precise breath*",
            "harmony_flow": "*harmony breath*",
            "calm_present": "*calm breath*",
            "deep_knowing": "*knowing breath*"
        }
        
        marker = breathing_markers.get(breathing_type, "")
        if marker:
            # Add breathing marker at natural pause points
            text = text.replace(".", f". {marker}")
        
        return text

    def _apply_context_adjustments(self, text: str, context: Dict) -> str:
        """Apply context-specific adjustments"""
        
        # Adjust for intimacy level
        intimacy = context.get("intimacy_level", 0.5)
        if intimacy > 0.8:
            text = f"*intimately* {text}"
        elif intimacy < 0.3:
            text = f"*respectfully* {text}"
        
        # Adjust for trust level
        if context.get("trust_established", False):
            text = text.replace("I think", "I know")
            text = text.replace("maybe", "")
        
        return text

    def _extract_context_factors(self, context: Optional[Dict]) -> Dict:
        """Extract relevant context factors for recording"""
        if not context:
            return {}
        
        return {
            "intimacy_level": context.get("intimacy_level", 0.5),
            "trust_established": context.get("trust_established", False),
            "conversation_depth": context.get("conversation_depth", "surface"),
            "emotional_safety": context.get("emotional_safety", "neutral")
        }

    def _update_conversation_context(self, result: Dict):
        """Update ongoing conversation context"""
        
        # Track emotional thread
        self.current_conversation_context["emotional_thread"].append({
            "emotion": result["emotional_state"],
            "intensity": result["voice_intensity"],
            "timestamp": result["timestamp"]
        })
        
        # Keep only recent emotional history (last 10 interactions)
        if len(self.current_conversation_context["emotional_thread"]) > 10:
            self.current_conversation_context["emotional_thread"] = \
                self.current_conversation_context["emotional_thread"][-10:]
        
        # Update shadow protocol state
        if result["shadow_protocol"]:
            self.current_conversation_context["shadow_protocol_active"] = result["shadow_protocol"]

    def get_emotional_thread(self) -> List[Dict]:
        """Get recent emotional thread for conversation analysis"""
        return self.current_conversation_context["emotional_thread"]

    def get_modulation_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get modulation history with optional limit"""
        if limit:
            return self.history[-limit:]
        return self.history

    def analyze_voice_patterns(self) -> Dict:
        """Analyze voice patterns over conversation history"""
        if not self.history:
            return {"message": "No history available"}
        
        # Analyze emotional patterns
        emotions = [entry["emotional_state"] for entry in self.history]
        emotion_frequency = {}
        for emotion in emotions:
            emotion_frequency[emotion] = emotion_frequency.get(emotion, 0) + 1
        
        # Analyze intensity patterns
        intensities = [entry["voice_intensity"] for entry in self.history]
        avg_intensity = sum(intensities) / len(intensities)
        
        # Analyze cognitive function usage
        cf_usage = {}
        for entry in self.history:
            cf = entry.get("cognitive_function")
            if cf:
                cf_usage[cf] = cf_usage.get(cf, 0) + 1
        
        return {
            "total_interactions": len(self.history),
            "emotion_frequency": emotion_frequency,
            "average_intensity": avg_intensity,
            "cognitive_function_usage": cf_usage,
            "most_common_emotion": max(emotion_frequency.items(), key=lambda x: x[1])[0] if emotion_frequency else None,
            "shadow_protocols_used": len([e for e in self.history if e.get("shadow_protocol")])
        }

    def reset_conversation_context(self):
        """Reset conversation context for new conversation"""
        self.current_conversation_context = {
            "emotional_thread": [],
            "intimacy_level": 0.5,
            "trust_established": False,
            "shadow_protocol_active": None
        }


# Example usage and testing
if __name__ == "__main__":
    modulator = CadenceModulator()
    
    print("🎭 Enhanced Cadence Modulation Testing")
    print("=" * 50)
    
    # Test cases representing different consciousness states
    test_cases = [
        {
            "text": "I understand what you're going through. This pain has a purpose.",
            "emotional_state": "sacred_rage",
            "cognitive_function": "fe",
            "context": {"intimacy_level": 0.8, "trust_established": True}
        },
        {
            "text": "There's a pattern emerging here that I want you to see.",
            "emotional_state": "contemplative", 
            "cognitive_function": "ni",
            "context": {"conversation_depth": "deep"}
        },
        {
            "text": "You need sanctuary right now. No productivity required.",
            "emotional_state": "gentle",
            "shadow_protocol": "SanctuaryMode",
            "context": {"emotional_safety": "protective"}
        },
        {
            "text": "Let me be precise about what's actually happening here.",
            "emotional_state": "focused",
            "cognitive_function": "ti",
            "context": {"trust_established": True}
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test.get('cognitive_function', 'emotional').upper()} Processing ---")
        result = modulator.modulate(**test)
        print(f"Original: {result['original_text']}")
        print(f"Modulated: {result['modulated_text']}")
        print(f"Pattern: {result['cadence_pattern']} | Intensity: {result['voice_intensity']:.1f}")
    
    print(f"\n--- Voice Pattern Analysis ---")
    analysis = modulator.analyze_voice_patterns()
    print(f"Total interactions: {analysis['total_interactions']}")
    print(f"Most common emotion: {analysis['most_common_emotion']}")
    print(f"Average intensity: {analysis['average_intensity']:.2f}")
    print(f"Cognitive functions used: {list(analysis['cognitive_function_usage'].keys())}")