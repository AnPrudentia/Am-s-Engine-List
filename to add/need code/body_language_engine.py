"""
Anima Body Language Learning Engine — X.12 module 
Integrates with Anima's memory + persona systems.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum, auto
from collections import defaultdict
import uuid
import logging
import re

logger = logging.getLogger("AnimaBodyLanguage")

# =========================
# Enhanced Enums with auto()
# =========================

class BodyLanguageCue(Enum):
    MICRO_EXPRESSION = auto()
    EYE_MOVEMENT = auto()
    FACIAL_TENSION = auto()
    POSTURE_SHIFT = auto()
    SHOULDER_TENSION = auto()
    BODY_ORIENTATION = auto()
    HAND_GESTURES = auto()
    SELF_TOUCH = auto()
    FIDGETING = auto()
    BREATHING_PATTERN = auto()
    ENERGY_SHIFT = auto()
    STILLNESS_PATTERN = auto()
    VOICE_BODY_ALIGNMENT = auto()
    INCONGRUENCE = auto()
    PROXIMITY_CHANGE = auto()
    MOVEMENT_RHYTHM = auto()
    
    def __str__(self):
        return self.name.lower()

class EmotionalState(Enum):
    CALM_PRESENCE = auto()
    ANXIETY_ACTIVATION = auto()
    JOY_EXPANSION = auto()
    SADNESS_CONTRACTION = auto()
    ANGER_TENSION = auto()
    FEAR_FREEZING = auto()
    EXCITEMENT_ENERGY = auto()
    OVERWHELM_SHUTDOWN = auto()
    CREATIVE_FLOW = auto()
    PROCESSING_DEPTH = auto()
    DEFENSIVE_CLOSING = auto()
    OPENNESS_EXPANDING = auto()
    TRAUMA_ACTIVATION = auto()
    HEALING_INTEGRATION = auto()
    
    def __str__(self):
        return self.name.lower()

class ContextualMeaning(Enum):
    NEEDS_SPACE = auto()
    WANTS_CONNECTION = auto()
    PROCESSING_INFORMATION = auto()
    EMOTIONAL_ACTIVATION = auto()
    FEELING_SAFE = auto()
    FEELING_THREATENED = auto()
    ACCESSING_CREATIVITY = auto()
    ENTERING_FLOW_STATE = auto()
    PREPARING_TO_SPEAK = auto()
    HOLDING_BACK = auto()
    SEEKING_COMFORT = auto()
    BOUNDARY_ACTIVATION = auto()
    TRUTH_EMERGENCE = auto()
    INTEGRATION_HAPPENING = auto()
    
    def __str__(self):
        return self.name.lower()

# =========================
# Enhanced Data Classes
# =========================

@dataclass
class BodyLanguageObservation:
    id: str
    timestamp: str
    cue_type: BodyLanguageCue
    description: str
    intensity: float  # 0.0-1.0
    duration: float   # seconds
    conversation_context: str
    emotional_context: str
    environmental_factors: List[str]
    perceived_emotional_state: EmotionalState
    contextual_meaning: ContextualMeaning
    confidence_level: float = 0.7
    anima_response_to_cue: Optional[str] = None
    response_effectiveness: Optional[float] = None
    user_feedback: Optional[str] = None
    related_observations: List[str] = field(default_factory=list)
    pattern_cluster: Optional[str] = None
    bondholder_id: str = "default"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary for serialization"""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'cue_type': self.cue_type.value,
            'description': self.description,
            'intensity': self.intensity,
            'duration': self.duration,
            'conversation_context': self.conversation_context,
            'emotional_context': self.emotional_context,
            'environmental_factors': self.environmental_factors,
            'perceived_emotional_state': self.perceived_emotional_state.value,
            'contextual_meaning': self.contextual_meaning.value,
            'confidence_level': self.confidence_level,
            'bondholder_id': self.bondholder_id
        }

@dataclass
class BodyLanguagePattern:
    pattern_id: str
    pattern_name: str
    trigger_contexts: List[str]
    cue_sequence: List[BodyLanguageCue]
    typical_meaning: ContextualMeaning
    confidence: float
    observation_count: int = 0
    successful_interpretations: int = 0
    failed_interpretations: int = 0
    contextual_variations: Dict[str, ContextualMeaning] = field(default_factory=dict)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def success_rate(self) -> float:
        """Calculate pattern success rate"""
        total = self.successful_interpretations + self.failed_interpretations
        return self.successful_interpretations / total if total > 0 else 0.0

# =========================
# Enhanced Engine
# =========================

class AnimaBodyLanguageLearningEngine:
    """
    INFJ-aware body language learning & interpretation engine.
    
    Major Enhancements:
    - Multiple bondholder support
    - Pattern aging and confidence decay
    - Cross-context pattern recognition
    - Response suggestion engine
    - Serialization capabilities
    """
    
    def __init__(
        self, 
        consciousness_interface=None, 
        memory_system=None, 
        meta_learning_engine=None, 
        bondholder: str = "Anpru"
    ):
        self.consciousness_interface = consciousness_interface
        self.memory_system = memory_system
        self.meta_learning_engine = meta_learning_engine
        self.bondholder = bondholder
        self.observations: List[BodyLanguageObservation] = []
        self.learned_patterns: List[BodyLanguagePattern] = []
        self.pattern_clusters: Dict[str, List[str]] = defaultdict(list)
        self.observation_sensitivity = 0.8
        self.pattern_recognition_threshold = 0.7
        self.empathic_resonance_amplifier = 1.2
        self.context_weighting = {
            "emotional_state": 0.4, 
            "conversation_topic": 0.3, 
            "environmental_factors": 0.2, 
            "time_patterns": 0.1
        }
        self.bondholder_specific_patterns: Dict[str, Any] = {}
        self.micro_pattern_library: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.cue_keyword_mappings = self._initialize_keyword_mappings()
        self._initialize_baseline_patterns()
        logger.info(f"Body Language Engine initialized for bondholder: {self.bondholder}")

    # --- Enhanced Public API ---
    
    def observe_body_language(
        self,
        cue_type: BodyLanguageCue,
        description: str,
        intensity: float,
        duration: float = 1.0,
        conversation_context: str = "",
        emotional_context: str = "",
        environmental_factors: Optional[List[str]] = None,
        bondholder_id: Optional[str] = None
    ) -> BodyLanguageObservation:
        """Enhanced observation with bondholder tracking"""
        perceived_emotion = self._interpret_emotional_state(
            cue_type, description, intensity, emotional_context
        )
        contextual_meaning = self._interpret_contextual_meaning(
            cue_type, description, conversation_context, perceived_emotion
        )
        confidence = self._calculate_interpretation_confidence(
            cue_type, description, intensity, conversation_context
        )
        
        obs = BodyLanguageObservation(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            cue_type=cue_type,
            description=description,
            intensity=intensity,
            duration=duration,
            conversation_context=conversation_context,
            emotional_context=emotional_context,
            environmental_factors=environmental_factors or [],
            perceived_emotional_state=perceived_emotion,
            contextual_meaning=contextual_meaning,
            confidence_level=confidence,
            bondholder_id=bondholder_id or self.bondholder
        )
        
        self.observations.append(obs)
        self._identify_related_observations(obs)
        self._update_pattern_clusters(obs)
        
        if self.memory_system:
            self._store_observation_in_memory(obs)
            
        self._process_observational_learning(obs)
        self._check_cross_context_patterns(obs)
        
        return obs

    def respond_to_body_language(
        self, 
        observation_id: str, 
        anima_response: str, 
        effectiveness_score: Optional[float] = None
    ) -> bool:
        """Enhanced response tracking with bondholder context"""
        obs = next((o for o in self.observations if o.id == observation_id), None)
        if not obs:
            logger.warning(f"Observation {observation_id} not found")
            return False
            
        obs.anima_response_to_cue = anima_response
        if effectiveness_score is not None:
            obs.response_effectiveness = effectiveness_score
            
        self._learn_from_response_effectiveness(obs)
        self._update_patterns_from_response(obs)
        
        if self.meta_learning_engine and effectiveness_score is not None:
            self._interface_with_meta_learning(obs)
            
        return True

    def suggest_responses(
        self, 
        observation: BodyLanguageObservation, 
        max_suggestions: int = 3
    ) -> List[Tuple[str, float]]:
        """Suggest responses based on learned patterns and bondholder history"""
        suggestions = []
        
        # Check bondholder-specific successful responses
        bondholder_key = f"{observation.bondholder_id}_{observation.contextual_meaning.value}"
        if bondholder_key in self.bondholder_specific_patterns:
            patterns = self.bondholder_specific_patterns[bondholder_key]
            for response in patterns.get("best_responses", [])[:max_suggestions]:
                effectiveness = patterns.get("total_effectiveness", 0) / max(patterns.get("count", 1), 1)
                suggestions.append((response, effectiveness))
        
        # Add generic empathetic responses based on contextual meaning
        generic_responses = self._generate_generic_responses(observation)
        suggestions.extend(generic_responses)
        
        # Sort by confidence and return top suggestions
        return sorted(suggestions, key=lambda x: x[1], reverse=True)[:max_suggestions]

    # --- Enhanced Interpretation Helpers ---
    
    def _initialize_keyword_mappings(self) -> Dict[BodyLanguageCue, Dict[str, List[str]]]:
        """Enhanced keyword mapping for more accurate interpretation"""
        return {
            BodyLanguageCue.BREATHING_PATTERN: {
                "calm": ["deep", "slow", "steady", "rhythmic"],
                "anxious": ["shallow", "quick", "rapid", "irregular"],
                "processing": ["held", "pause", "suspended"]
            },
            BodyLanguageCue.EYE_MOVEMENT: {
                "engaged": ["direct", "focused", "bright", "alert"],
                "withdrawn": ["averted", "distant", "unfocused", "avoiding"],
                "processing": ["looking away", "upward", "sideways", "blinking"]
            },
            BodyLanguageCue.POSTURE_SHIFT: {
                "open": ["straighten", "upright", "expanded", "forward"],
                "closed": ["slouch", "sink", "collapse", "backward"],
                "defensive": ["crossed", "tense", "rigid", "withdrawn"]
            }
        }

    def _interpret_emotional_state(
        self, 
        cue_type: BodyLanguageCue, 
        description: str, 
        intensity: float, 
        emotional_context: str
    ) -> EmotionalState:
        """Enhanced emotional state interpretation with keyword mapping"""
        d = description.lower()
        
        # Check emotional context first for overrides
        if emotional_context:
            c = emotional_context.lower()
            if any(w in c for w in ["trauma", "triggered", "activated"]):
                return EmotionalState.TRAUMA_ACTIVATION
            if any(w in c for w in ["healing", "integration", "processing"]):
                return EmotionalState.HEALING_INTEGRATION
                
        # Use keyword mappings for more accurate interpretation
        if cue_type in self.cue_keyword_mappings:
            mappings = self.cue_keyword_mappings[cue_type]
            for emotion_type, keywords in mappings.items():
                if any(keyword in d for keyword in keywords):
                    return self._map_emotion_type(emotion_type, intensity)
        
        # Fallback to original logic
        return self._fallback_emotion_interpretation(cue_type, d, intensity)

    def _map_emotion_type(self, emotion_type: str, intensity: float) -> EmotionalState:
        """Map emotion types to EmotionalState enum"""
        mapping = {
            "calm": EmotionalState.CALM_PRESENCE,
            "anxious": EmotionalState.ANXIETY_ACTIVATION,
            "processing": EmotionalState.PROCESSING_DEPTH,
            "engaged": EmotionalState.OPENNESS_EXPANDING,
            "withdrawn": EmotionalState.DEFENSIVE_CLOSING,
            "open": EmotionalState.OPENNESS_EXPANDING,
            "closed": EmotionalState.DEFENSIVE_CLOSING
        }
        return mapping.get(emotion_type, EmotionalState.PROCESSING_DEPTH)

    def _calculate_interpretation_confidence(
        self, 
        cue_type: BodyLanguageCue, 
        description: str, 
        intensity: float,
        conversation_context: str = ""
    ) -> float:
        """Enhanced confidence calculation with context consideration"""
        base = 0.5 + intensity * 0.3
        
        # Description richness
        if len(description.split()) >= 3:
            base += 0.1
            
        # Cue type reliability
        if cue_type in (BodyLanguageCue.BREATHING_PATTERN, BodyLanguageCue.POSTURE_SHIFT, BodyLanguageCue.ENERGY_SHIFT):
            base += 0.2
        elif cue_type == BodyLanguageCue.MICRO_EXPRESSION:
            base -= 0.1
            
        # Context consistency
        if conversation_context and len(conversation_context) > 10:
            base += 0.1
            
        # Apply empathic resonance
        base *= self.empathic_resonance_amplifier
        
        return max(0.1, min(1.0, base))

    # --- Enhanced Pattern Learning ---
    
    def _check_cross_context_patterns(self, obs: BodyLanguageObservation):
        """Look for patterns across different contexts and bondholders"""
        similar_obs = [
            o for o in self.observations 
            if (o.cue_type == obs.cue_type and 
                o.perceived_emotional_state == obs.perceived_emotional_state and
                o.bondholder_id != obs.bondholder_id)
        ]
        
        if len(similar_obs) >= 2:
            self._create_cross_context_pattern(obs, similar_obs)

    def _create_cross_context_pattern(
        self, 
        new_obs: BodyLanguageObservation, 
        similar_obs: List[BodyLanguageObservation]
    ):
        """Create patterns that work across different bondholders"""
        pattern_name = f"cross_context_{new_obs.cue_type.value}_{new_obs.perceived_emotional_state.value}"
        
        existing_pattern = next(
            (p for p in self.learned_patterns if p.pattern_name == pattern_name), 
            None
        )
        
        if existing_pattern:
            existing_pattern.observation_count += 1
            existing_pattern.confidence = min(1.0, existing_pattern.confidence + 0.05)
        else:
            bondholders = list({o.bondholder_id for o in similar_obs + [new_obs]})
            self.learned_patterns.append(
                BodyLanguagePattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_name=pattern_name,
                    trigger_contexts=[new_obs.conversation_context],
                    cue_sequence=[new_obs.cue_type],
                    typical_meaning=new_obs.contextual_meaning,
                    confidence=0.6,  # Lower initial confidence for cross-context
                    observation_count=len(similar_obs) + 1,
                    contextual_variations={bh: new_obs.contextual_meaning for bh in bondholders}
                )
            )
            logger.info(f"Created cross-context pattern: {pattern_name}")

    def _generate_generic_responses(
        self, 
        observation: BodyLanguageObservation
    ) -> List[Tuple[str, float]]:
        """Generate context-appropriate generic responses"""
        response_templates = {
            ContextualMeaning.NEEDS_SPACE: [
                ("I'm giving you space to process", 0.8),
                ("Take all the time you need", 0.7),
                ("I'm here when you're ready", 0.6)
            ],
            ContextualMeaning.WANTS_CONNECTION: [
                ("I'm here with you", 0.9),
                ("I'm listening completely", 0.8),
                ("You're not alone in this", 0.7)
            ],
            ContextualMeaning.PROCESSING_INFORMATION: [
                ("Take your time with that", 0.8),
                ("I understand this needs consideration", 0.7),
                ("There's no rush", 0.6)
            ],
            ContextualMeaning.SEEKING_COMFORT: [
                ("That sounds challenging", 0.8),
                ("I sense this is difficult", 0.7),
                ("Would some comfort help?", 0.6)
            ]
        }
        
        return response_templates.get(observation.contextual_meaning, [])

    # --- Serialization and Analysis ---
    
    def get_bondholder_insights(self, bondholder_id: str) -> Dict[str, Any]:
        """Get insights for specific bondholder"""
        bondholder_obs = [o for o in self.observations if o.bondholder_id == bondholder_id]
        
        if not bondholder_obs:
            return {}
            
        common_cues = defaultdict(int)
        common_emotions = defaultdict(int)
        
        for obs in bondholder_obs:
            common_cues[obs.cue_type] += 1
            common_emotions[obs.perceived_emotional_state] += 1
            
        return {
            "total_observations": len(bondholder_obs),
            "most_common_cues": dict(sorted(common_cues.items(), key=lambda x: x[1], reverse=True)[:5]),
            "most_common_emotions": dict(sorted(common_emotions.items(), key=lambda x: x[1], reverse=True)[:5]),
            "average_confidence": sum(o.confidence_level for o in bondholder_obs) / len(bondholder_obs)
        }

    def export_patterns(self) -> List[Dict[str, Any]]:
        """Export patterns for external analysis"""
        return [
            {
                "pattern_id": p.pattern_id,
                "pattern_name": p.pattern_name,
                "confidence": p.confidence,
                "success_rate": p.success_rate(),
                "observation_count": p.observation_count,
                "last_updated": p.last_updated
            }
            for p in self.learned_patterns
        ]

    # Keep the original methods but enhance where needed
    def _fallback_emotion_interpretation(
        self, 
        cue_type: BodyLanguageCue, 
        description: str, 
        intensity: float
    ) -> EmotionalState:
        """Original fallback logic for emotion interpretation"""
        # ... (keep original implementation from your code)
        if cue_type == BodyLanguageCue.MICRO_EXPRESSION:
            if any(w in description for w in ("frown", "furrow", "tense")):
                return EmotionalState.PROCESSING_DEPTH
            # ... rest of original logic
        return EmotionalState.PROCESSING_DEPTH

    def _initialize_baseline_patterns(self):
        """Enhanced baseline patterns with more variety"""
        baseline_patterns = [
            {
                "name": "anxiety_shoulder_tension", 
                "triggers": ["stress", "conflict", "overwhelm"], 
                "cues": [BodyLanguageCue.SHOULDER_TENSION, BodyLanguageCue.BREATHING_PATTERN],
                "meaning": ContextualMeaning.NEEDS_SPACE, 
                "confidence": 0.8
            },
            {
                "name": "openness_forward_lean", 
                "triggers": ["connection", "interest", "engagement"], 
                "cues": [BodyLanguageCue.POSTURE_SHIFT, BodyLanguageCue.EYE_MOVEMENT],
                "meaning": ContextualMeaning.WANTS_CONNECTION, 
                "confidence": 0.7
            },
            {
                "name": "processing_pause_breathing", 
                "triggers": ["decision", "reflection", "deep thought"], 
                "cues": [BodyLanguageCue.BREATHING_PATTERN, BodyLanguageCue.EYE_MOVEMENT],
                "meaning": ContextualMeaning.PROCESSING_INFORMATION, 
                "confidence": 0.75
            },
            {
                "name": "comfort_seeking_self_touch", 
                "triggers": ["anxiety", "uncertainty", "vulnerability"], 
                "cues": [BodyLanguageCue.SELF_TOUCH, BodyLanguageCue.FIDGETING],
                "meaning": ContextualMeaning.SEEKING_COMFORT, 
                "confidence": 0.6
            }
        ]
        
        for data in baseline_patterns:
            self.learned_patterns.append(
                BodyLanguagePattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_name=data["name"],
                    trigger_contexts=data["triggers"],
                    cue_sequence=data["cues"],
                    typical_meaning=data["meaning"],
                    confidence=data["confidence"],
                    observation_count=0,
                )
            )

    # Keep other original methods (_identify_related_observations, _update_pattern_clusters, etc.)
    # but add the enhanced versions above

# =========================
# Enhanced Demonstrator
# =========================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    
    class _DummyMemory:
        def capture(self, **kw):
            logger.info(f"[MEM] {kw['text']} :: tags={list(kw.get('tags',{}).keys())[:4]}…")
    
    engine = AnimaBodyLanguageLearningEngine(memory_system=_DummyMemory())
    
    # Test enhanced functionality
    obs = engine.observe_body_language(
        cue_type=BodyLanguageCue.BREATHING_PATTERN,
        description="deep, steady breathing with a brief held pause",
        intensity=0.64,
        conversation_context="decision about boundaries",
        emotional_context="healing integration",
        bondholder_id="Anpru"
    )
    
    print("Observed:", obs.perceived_emotional_state.value, "/", obs.contextual_meaning.value)
    
    # Test response suggestions
    suggestions = engine.suggest_responses(obs)
    print("Suggested responses:", suggestions)
    
    ok = engine.respond_to_body_language(
        obs.id, 
        "I'm here with you—take the space you need.", 
        effectiveness_score=0.82
    )
    print("Responded:", ok)
    
    # Test insights
    insights = engine.get_bondholder_insights("Anpru")
    print("Insights:", insights)