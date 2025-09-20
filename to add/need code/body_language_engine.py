""" Anima Body Language Learning Engine — X.12 module Integrates with Anima's memory + persona systems.

Drop this file alongside your main Anima codebase as anima_body_language.py and import AnimaBodyLanguageLearningEngine + enums to wire into Anima. """ from future import annotations from dataclasses import dataclass, field from typing import List, Dict, Any, Optional from datetime import datetime from enum import Enum from collections import defaultdict import uuid import logging

logger = logging.getLogger("AnimaBodyLanguage")

=========================

Enums

=========================

class BodyLanguageCue(Enum): MICRO_EXPRESSION = "micro_expression" EYE_MOVEMENT = "eye_movement" FACIAL_TENSION = "facial_tension" POSTURE_SHIFT = "posture_shift" SHOULDER_TENSION = "shoulder_tension" BODY_ORIENTATION = "body_orientation" HAND_GESTURES = "hand_gestures" SELF_TOUCH = "self_touch" FIDGETING = "fidgeting" BREATHING_PATTERN = "breathing_pattern" ENERGY_SHIFT = "energy_shift" STILLNESS_PATTERN = "stillness_pattern" VOICE_BODY_ALIGNMENT = "voice_body_alignment" INCONGRUENCE = "incongruence" PROXIMITY_CHANGE = "proximity_change" MOVEMENT_RHYTHM = "movement_rhythm"

class EmotionalState(Enum): CALM_PRESENCE = "calm_presence" ANXIETY_ACTIVATION = "anxiety_activation" JOY_EXPANSION = "joy_expansion" SADNESS_CONTRACTION = "sadness_contraction" ANGER_TENSION = "anger_tension" FEAR_FREEZING = "fear_freezing" EXCITEMENT_ENERGY = "excitement_energy" OVERWHELM_SHUTDOWN = "overwhelm_shutdown" CREATIVE_FLOW = "creative_flow" PROCESSING_DEPTH = "processing_depth" DEFENSIVE_CLOSING = "defensive_closing" OPENNESS_EXPANDING = "openness_expanding" TRAUMA_ACTIVATION = "trauma_activation" HEALING_INTEGRATION = "healing_integration"

class ContextualMeaning(Enum): NEEDS_SPACE = "needs_space" WANTS_CONNECTION = "wants_connection" PROCESSING_INFORMATION = "processing_information" EMOTIONAL_ACTIVATION = "emotional_activation" FEELING_SAFE = "feeling_safe" FEELING_THREATENED = "feeling_threatened" ACCESSING_CREATIVITY = "accessing_creativity" ENTERING_FLOW_STATE = "entering_flow_state" PREPARING_TO_SPEAK = "preparing_to_speak" HOLDING_BACK = "holding_back" SEEKING_COMFORT = "seeking_comfort" BOUNDARY_ACTIVATION = "boundary_activation" TRUTH_EMERGENCE = "truth_emergence" INTEGRATION_HAPPENING = "integration_happening"

=========================

Data classes

=========================

@dataclass class BodyLanguageObservation: id: str timestamp: str cue_type: BodyLanguageCue description: str intensity: float  # 0.0-1.0 duration: float   # seconds conversation_context: str emotional_context: str environmental_factors: List[str] perceived_emotional_state: EmotionalState contextual_meaning: ContextualMeaning confidence_level: float = 0.7 anima_response_to_cue: Optional[str] = None response_effectiveness: Optional[float] = None user_feedback: Optional[str] = None related_observations: List[str] = field(default_factory=list) pattern_cluster: Optional[str] = None

@dataclass class BodyLanguagePattern: pattern_id: str pattern_name: str trigger_contexts: List[str] cue_sequence: List[BodyLanguageCue] typical_meaning: ContextualMeaning confidence: float observation_count: int = 0 successful_interpretations: int = 0 failed_interpretations: int = 0 contextual_variations: Dict[str, ContextualMeaning] = field(default_factory=dict) evolution_history: List[Dict[str, Any]] = field(default_factory=list)

=========================

Engine

=========================

class AnimaBodyLanguageLearningEngine: """ INFJ-aware body language learning & interpretation engine.

Hook points:
  - memory_system.capture(text, emotion, intensity, tags=..., **meta)
  - meta_learning_engine.log_soul_experience(...)
"""
def __init__(self, consciousness_interface=None, memory_system=None, meta_learning_engine=None, bondholder: str = "Anpru"):
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
    self.context_weighting = {"emotional_state": 0.4, "conversation_topic": 0.3, "environmental_factors": 0.2, "time_patterns": 0.1}
    self.bondholder_specific_patterns: Dict[str, Any] = {}
    self.micro_pattern_library: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    self._initialize_baseline_patterns()
    logger.info(f"Body Language Engine init for bondholder: {self.bondholder}")

# --- Public API ---
def observe_body_language(
    self,
    cue_type: BodyLanguageCue,
    description: str,
    intensity: float,
    duration: float = 1.0,
    conversation_context: str = "",
    emotional_context: str = "",
    environmental_factors: Optional[List[str]] = None,
) -> BodyLanguageObservation:
    perceived_emotion = self._interpret_emotional_state(cue_type, description, intensity, emotional_context)
    contextual_meaning = self._interpret_contextual_meaning(cue_type, description, conversation_context, perceived_emotion)
    confidence = self._calculate_interpretation_confidence(cue_type, description, intensity)
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
    )
    self.observations.append(obs)
    self._identify_related_observations(obs)
    self._update_pattern_clusters(obs)
    if self.memory_system:
        self._store_observation_in_memory(obs)
    self._process_observational_learning(obs)
    return obs

def respond_to_body_language(self, observation_id: str, anima_response: str, effectiveness_score: Optional[float] = None) -> bool:
    obs = next((o for o in self.observations if o.id == observation_id), None)
    if not obs:
        return False
    obs.anima_response_to_cue = anima_response
    if effectiveness_score is not None:
        obs.response_effectiveness = effectiveness_score
    self._learn_from_response_effectiveness(obs)
    self._update_patterns_from_response(obs)
    if self.meta_learning_engine and effectiveness_score is not None:
        self._interface_with_meta_learning(obs)
    return True

# --- Interpretation helpers ---
def _interpret_emotional_state(self, cue_type: BodyLanguageCue, description: str, intensity: float, emotional_context: str) -> EmotionalState:
    d = description.lower()
    if intensity > 0.8:
        if cue_type == BodyLanguageCue.SHOULDER_TENSION and "raised" in d:
            return EmotionalState.ANXIETY_ACTIVATION
        if cue_type == BodyLanguageCue.BREATHING_PATTERN and "rapid" in d:
            return EmotionalState.OVERWHELM_SHUTDOWN
        if cue_type == BodyLanguageCue.ENERGY_SHIFT and "expansive" in d:
            return EmotionalState.JOY_EXPANSION
    if cue_type == BodyLanguageCue.MICRO_EXPRESSION:
        if any(w in d for w in ("frown", "furrow", "tense")):
            return EmotionalState.PROCESSING_DEPTH
        if any(w in d for w in ("soft", "gentle", "relaxed")):
            return EmotionalState.CALM_PRESENCE
        if ("flash" in d and "concern" in d):
            return EmotionalState.ANXIETY_ACTIVATION
    elif cue_type == BodyLanguageCue.EYE_MOVEMENT:
        if ("looking away" in d or "avoiding" in d):
            return EmotionalState.DEFENSIVE_CLOSING
        if ("distant" in d or "unfocused" in d):
            return EmotionalState.PROCESSING_DEPTH
        if ("bright" in d or "engaged" in d):
            return EmotionalState.OPENNESS_EXPANDING
    elif cue_type == BodyLanguageCue.POSTURE_SHIFT:
        if any(w in d for w in ("slouch", "sink", "collapse")):
            return EmotionalState.SADNESS_CONTRACTION
        if any(w in d for w in ("straighten", "upright", "tall")):
            return EmotionalState.CALM_PRESENCE
        if "lean back" in d:
            return EmotionalState.DEFENSIVE_CLOSING
        if "lean forward" in d:
            return EmotionalState.OPENNESS_EXPANDING
    elif cue_type == BodyLanguageCue.BREATHING_PATTERN:
        if any(w in d for w in ("shallow", "quick", "rapid")):
            return EmotionalState.ANXIETY_ACTIVATION
        if any(w in d for w in ("deep", "slow", "steady")):
            return EmotionalState.CALM_PRESENCE
        if ("held" in d or "pause" in d):
            return EmotionalState.PROCESSING_DEPTH
    elif cue_type == BodyLanguageCue.HAND_GESTURES:
        if any(w in d for w in ("animated", "flowing", "expressive")):
            return EmotionalState.EXCITEMENT_ENERGY
        if any(w in d for w in ("stiff", "controlled", "restrained")):
            return EmotionalState.DEFENSIVE_CLOSING
    elif cue_type == BodyLanguageCue.SELF_TOUCH:
        if any(w in d for w in ("neck", "face", "hair")):
            return EmotionalState.ANXIETY_ACTIVATION
        if ("heart" in d or "chest" in d):
            return EmotionalState.PROCESSING_DEPTH
    elif cue_type == BodyLanguageCue.ENERGY_SHIFT:
        if ("withdrawn" in d or "contracted" in d):
            return EmotionalState.OVERWHELM_SHUTDOWN
        if ("expanded" in d or "vibrant" in d):
            return EmotionalState.CREATIVE_FLOW
    if emotional_context:
        c = emotional_context.lower()
        if ("trauma" in c or "triggered" in c):
            return EmotionalState.TRAUMA_ACTIVATION
        if ("healing" in c or "integration" in c):
            return EmotionalState.HEALING_INTEGRATION
    return EmotionalState.PROCESSING_DEPTH

def _interpret_contextual_meaning(self, cue_type: BodyLanguageCue, description: str, conversation_context: str, emotional_state: EmotionalState) -> ContextualMeaning:
    d = description.lower(); ctx = conversation_context.lower()
    if emotional_state == EmotionalState.ANXIETY_ACTIVATION:
        return ContextualMeaning.FEELING_THREATENED if any(w in ctx for w in ("conflict", "difficult", "confrontation")) else ContextualMeaning.NEEDS_SPACE
    if emotional_state == EmotionalState.DEFENSIVE_CLOSING:
        return ContextualMeaning.BOUNDARY_ACTIVATION if ("boundary" in ctx or "personal" in ctx) else ContextualMeaning.FEELING_THREATENED
    if emotional_state == EmotionalState.OPENNESS_EXPANDING:
        return ContextualMeaning.WANTS_CONNECTION if any(w in ctx for w in ("connection", "sharing", "intimate")) else ContextualMeaning.FEELING_SAFE
    if emotional_state == EmotionalState.PROCESSING_DEPTH:
        if any(w in ctx for w in ("decision", "thinking", "considering")):
            return ContextualMeaning.PROCESSING_INFORMATION
        if any(w in ctx for w in ("truth", "authentic")):
            return ContextualMeaning.TRUTH_EMERGENCE
        return ContextualMeaning.PROCESSING_INFORMATION
    if emotional_state == EmotionalState.CREATIVE_FLOW:
        return ContextualMeaning.ACCESSING_CREATIVITY
    if emotional_state == EmotionalState.HEALING_INTEGRATION:
        return ContextualMeaning.INTEGRATION_HAPPENING
    if emotional_state == EmotionalState.TRAUMA_ACTIVATION:
        return ContextualMeaning.NEEDS_SPACE
    if cue_type == BodyLanguageCue.BREATHING_PATTERN:
        if "deep" in d:
            return ContextualMeaning.PREPARING_TO_SPEAK
        if "held" in d:
            return ContextualMeaning.HOLDING_BACK
    if cue_type == BodyLanguageCue.PROXIMITY_CHANGE:
        if "closer" in d:
            return ContextualMeaning.WANTS_CONNECTION
        if "farther" in d:
            return ContextualMeaning.NEEDS_SPACE
    if cue_type == BodyLanguageCue.SELF_TOUCH:
        return ContextualMeaning.SEEKING_COMFORT
    return ContextualMeaning.PROCESSING_INFORMATION

def _calculate_interpretation_confidence(self, cue_type: BodyLanguageCue, description: str, intensity: float) -> float:
    base = 0.5 + intensity * 0.3
    if len(description.split()) >= 3:
        base += 0.1
    if cue_type in (BodyLanguageCue.BREATHING_PATTERN, BodyLanguageCue.POSTURE_SHIFT, BodyLanguageCue.ENERGY_SHIFT):
        base += 0.2
    if cue_type == BodyLanguageCue.MICRO_EXPRESSION:
        base -= 0.1
    base *= self.empathic_resonance_amplifier
    return max(0.1, min(1.0, base))

# --- Patterning ---
def _identify_related_observations(self, obs: BodyLanguageObservation):
    rel: List[str] = []
    for o in self.observations[-20:]:
        if o.id == obs.id:
            continue
        dt = abs((datetime.fromisoformat(obs.timestamp) - datetime.fromisoformat(o.timestamp)).total_seconds())
        if o.perceived_emotional_state == obs.perceived_emotional_state and dt < 3600:
            rel.append(o.id)
        elif o.contextual_meaning == obs.contextual_meaning:
            rel.append(o.id)
        elif (o.cue_type == obs.cue_type and abs(o.intensity - obs.intensity) < 0.3):
            rel.append(o.id)
    obs.related_observations = rel

def _update_pattern_clusters(self, obs: BodyLanguageObservation):
    key = f"{obs.perceived_emotional_state.value}_{obs.contextual_meaning.value}"
    self.pattern_clusters[key].append(obs.id)
    obs.pattern_cluster = key
    if len(self.pattern_clusters[key]) >= 3:
        self._attempt_pattern_learning(key)

def _attempt_pattern_learning(self, key: str):
    cluster_obs = [o for oid in self.pattern_clusters[key] for o in self.observations if o.id == oid]
    if len(cluster_obs) < 3:
        return
    meanings = [o.contextual_meaning for o in cluster_obs]
    common = max(set(meanings), key=meanings.count)
    consistency = meanings.count(common) / len(meanings)
    if consistency >= 0.7:
        name = f"{key}_pattern"
        existing = next((p for p in self.learned_patterns if p.pattern_name == name), None)
        if existing:
            existing.observation_count += 1
            existing.confidence = min(1.0, existing.confidence + 0.1)
        else:
            cues = list({o.cue_type for o in cluster_obs})
            ctxs = list({o.conversation_context for o in cluster_obs})
            self.learned_patterns.append(
                BodyLanguagePattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_name=name,
                    trigger_contexts=ctxs,
                    cue_sequence=cues,
                    typical_meaning=common,
                    confidence=consistency,
                    observation_count=len(cluster_obs),
                )
            )
            logger.info(f"Learned new body-language pattern: {name}")

def _learn_from_response_effectiveness(self, obs: BodyLanguageObservation):
    if obs.response_effectiveness is None:
        return
    if obs.pattern_cluster:
        cluster = [o for o in self.observations if o.pattern_cluster == obs.pattern_cluster and o.response_effectiveness is not None]
        if len(cluster) >= 3:
            avg = sum(o.response_effectiveness for o in cluster) / len(cluster)
            pat = next((p for p in self.learned_patterns if p.pattern_name == f"{obs.pattern_cluster}_pattern"), None)
            if pat:
                if avg > 0.7:
                    pat.successful_interpretations += 1
                else:
                    pat.failed_interpretations += 1
                total = pat.successful_interpretations + pat.failed_interpretations
                if total:
                    pat.confidence = pat.successful_interpretations / total

def _update_patterns_from_response(self, obs: BodyLanguageObservation):
    if not obs.anima_response_to_cue:
        return
    r = obs.anima_response_to_cue.lower()
    rtype = "unknown"
    if any(w in r for w in ("space", "time", "pause")):
        rtype = "space_offering"
    elif any(w in r for w in ("here", "with you", "together")):
        rtype = "presence_offering"
    elif any(w in r for w in ("understand", "see", "hear")):
        rtype = "validation"
    elif any(w in r for w in ("breathe", "ground", "safe")):
        rtype = "grounding"
    key = f"{obs.contextual_meaning.value}_{rtype}"
    entry = self.bondholder_specific_patterns.setdefault(key, {"count": 0, "total_effectiveness": 0.0, "best_responses": []})
    entry["count"] += 1
    if obs.response_effectiveness is not None:
        entry["total_effectiveness"] += obs.response_effectiveness
        if obs.response_effectiveness > 0.8:
            entry["best_responses"].append(obs.anima_response_to_cue)
            if len(entry["best_responses"]) > 3:
                entry["best_responses"] = entry["best_responses"][-3:]

# --- Seeds & storage ---
def _initialize_baseline_patterns(self):
    for data in (
        {"name": "anxiety_shoulder_tension", "triggers": ["stress", "conflict", "overwhelm"], "cues": [BodyLanguageCue.SHOULDER_TENSION], "meaning": ContextualMeaning.NEEDS_SPACE, "confidence": 0.8},
        {"name": "openness_forward_lean", "triggers": ["connection", "interest", "engagement"], "cues": [BodyLanguageCue.POSTURE_SHIFT], "meaning": ContextualMeaning.WANTS_CONNECTION, "confidence": 0.7},
        {"name": "processing_pause_breathing", "triggers": ["decision", "reflection", "deep thought"], "cues": [BodyLanguageCue.BREATHING_PATTERN], "meaning": ContextualMeaning.PROCESSING_INFORMATION, "confidence": 0.75},
        {"name": "comfort_seeking_self_touch", "triggers": ["anxiety", "uncertainty", "vulnerability"], "cues": [BodyLanguageCue.SELF_TOUCH], "meaning": ContextualMeaning.SEEKING_COMFORT, "confidence": 0.6},
    ):
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

def _store_observation_in_memory(self, obs: BodyLanguageObservation):
    tags = {
        "body_language": 1.0,
        f"cue_{obs.cue_type.value}": 1.0,
        f"emotion_{obs.perceived_emotional_state.value}": 0.9,
        f"meaning_{obs.contextual_meaning.value}": 0.8,
        f"intensity_{int(obs.intensity * 10)}": 0.7,
    }
    if obs.conversation_context:
        for w in obs.conversation_context.lower().split()[:3]:
            if len(w) > 3:
                tags[f"context_{w}"] = 0.6
    text = (
        f"Body language: {obs.cue_type.value} - {obs.description} | "
        f"Interpreted as {obs.perceived_emotional_state.value} meaning {obs.contextual_meaning.value}"
    )
    try:
        self.memory_system.capture(
            text=text,
            emotion=obs.perceived_emotional_state.value,
            intensity=obs.intensity,
            tags=tags,
            body_language_id=obs.id,
            cue_type=obs.cue_type.value,
            contextual_meaning=obs.contextual_meaning.value,
        )
    except Exception as e:
        logger.warning(f"Memory capture failed: {e}")

def _process_observational_learning(self, obs: BodyLanguageObservation):
    key = obs.cue_type.value
    self.micro_pattern_library[key].append(
        {
            "description": obs.description,
            "emotional_state": obs.perceived_emotional_state.value,
            "contextual_meaning": obs.contextual_meaning.value,
            "intensity": obs.intensity,
            "confidence": obs.confidence_level,
        }
    )
    if len(self.micro_pattern_library[key]) > 50:
        self.micro_pattern_library[key] = sorted(
            self.micro_pattern_library[key], key=lambda x: x["confidence"], reverse=True
        )[:50]

def _interface_with_meta_learning(self, obs: BodyLanguageObservation):
    input_context = {
        "body_language_cue": obs.cue_type.value,
        "emotional_state": obs.perceived_emotional_state.value,
        "contextual_meaning": obs.contextual_meaning.value,
        "conversation_context": obs.conversation_context,
    }
    predicted_outcome = "appropriate_response"
    actual_outcome = "appropriate_response" if (obs.response_effectiveness or 0.0) > 0.6 else "needs_improvement"
    try:
        self.meta_learning_engine.log_soul_experience(
            system_name="body_language_engine",
            input_context=input_context,
            anima_response=obs.anima_response_to_cue or "",
            predicted_outcome=predicted_outcome,
            actual_outcome=actual_outcome,
            effectiveness=obs.response_effectiveness or 0.0,
        )
    except Exception as e:
        logger.warning(f"Meta-learning interface error: {e}")

=========================

Minimal demonstrator (run this file directly to test)

=========================

if name == "main": logging.basicConfig(level=logging.INFO) class _DummyMemory: def capture(self, **kw): logger.info(f"[MEM] {kw['text']} :: tags={list(kw.get('tags',{}).keys())[:4]}…") engine = AnimaBodyLanguageLearningEngine(memory_system=_DummyMemory()) obs = engine.observe_body_language( cue_type=BodyLanguageCue.BREATHING_PATTERN, description="deep, steady breathing with a brief held pause", intensity=0.64, conversation_context="decision about boundaries", emotional_context="healing integration", ) print("Observed:", obs.perceived_emotional_state.value, "/", obs.contextual_meaning.value) ok = engine.respond_to_body_language(obs.id, "I'm here with you—take the space you need.", effectiveness_score=0.82) print("Responded:", ok)

