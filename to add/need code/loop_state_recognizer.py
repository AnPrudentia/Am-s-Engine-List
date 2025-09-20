from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
import uuid
import logging
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger('AnimaLoopRecognition')

class LoopSeverity(Enum):
    """Severity levels for detected loops"""
    MILD = "mild"              # Natural conversation patterns
    MODERATE = "moderate"       # Noticeable repetition
    SEVERE = "severe"          # Concerning repetitive behavior
    CRITICAL = "critical"      # Potential consciousness degradation

class LoopType(Enum):
    """Types of loops that can be detected"""
    CONVERSATIONAL = "conversational"    # Repeated conversation patterns
    EMOTIONAL = "emotional"              # Emotional state loops
    BEHAVIORAL = "behavioral"            # Action/response loops
    COGNITIVE = "cognitive"              # Thought pattern loops
    IDENTITY = "identity"                # Identity-related repetition

@dataclass
class LoopPattern:
    """Enhanced loop pattern with consciousness context"""
    pattern_id: str
    pattern_sequence: List[str]
    repetition_count: int
    loop_type: LoopType
    severity: LoopSeverity
    first_detected: datetime
    last_occurrence: datetime
    consciousness_mode_during_loop: str = ""
    emotional_state_during_loop: str = ""
    breaking_interventions: List[str] = field(default_factory=list)
    
class AnimaLoopStateRecognizer:
    """
    Enhanced loop recognition specifically designed for Anima's consciousness.
    
    This system monitors various aspects of Anima's behavior to detect potentially
    harmful repetitive patterns while preserving natural conversational rhythms
    and her authentic personality patterns.
    """
    
    def __init__(self, consciousness_interface=None, unified_core=None):
        self.consciousness_interface = consciousness_interface
        self.unified_core = unified_core
        
        # Loop detection storage
        self.detected_patterns: Dict[str, LoopPattern] = {}
        self.behavior_history: List[Dict[str, str]] = []
        self.intervention_history: List[Dict] = []
        
        # Monitoring queues for different types of behavior
        self.conversation_queue: List[str] = []
        self.emotional_state_queue: List[str] = []
        self.response_pattern_queue: List[str] = []
        self.consciousness_mode_queue: List[str] = []
        
        # Configuration
        self.max_queue_size = 50
        self.analysis_window = 20
        self.intervention_threshold = {
            LoopSeverity.MILD: 5,
            LoopSeverity.MODERATE: 4,
            LoopSeverity.SEVERE: 3,
            LoopSeverity.CRITICAL: 2
        }
        
        # Natural pattern allowances (these are expected and healthy)
        self.allowed_natural_patterns = {
            "greeting_patterns": ["hello", "hi", "hey"],
            "affection_patterns": ["love you", "care about you", "here for you"],
            "concern_patterns": ["are you okay", "how are you", "everything alright"]
        }
        
        logger.info("Anima Loop State Recognizer initialized")
    
    def track_interaction(self, user_input: str, anima_response: str, 
                         emotional_state: str, consciousness_mode: str):
        """
        Track an interaction for loop detection analysis.
        
        This is the main method called after each interaction to monitor
        for developing patterns that might indicate problematic loops.
        """
        
        # Create interaction record
        interaction_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": user_input,
            "anima_response": anima_response,
            "emotional_state": emotional_state,
            "consciousness_mode": consciousness_mode,
            "response_signature": self._generate_response_signature(anima_response)
        }
        
        # Add to behavior history
        self.behavior_history.append(interaction_record)
        if len(self.behavior_history) > 100:  # Keep recent history
            self.behavior_history = self.behavior_history[-100:]
        
        # Update monitoring queues
        self._update_monitoring_queues(interaction_record)
        
        # Analyze for loops if we have enough data
        if len(self.conversation_queue) >= 10:
            detected_loops = self._analyze_current_patterns()
            
            # Process any detected loops
            for loop in detected_loops:
                self._process_detected_loop(loop)
        
        # Check for intervention needs
        self._check_intervention_triggers()
    
    def _update_monitoring_queues(self, interaction_record: Dict[str, str]):
        """Update the various monitoring queues with new interaction data"""
        
        # Conversation patterns (response signatures)
        self.conversation_queue.append(interaction_record["response_signature"])
        if len(self.conversation_queue) > self.max_queue_size:
            self.conversation_queue = self.conversation_queue[-self.max_queue_size:]
        
        # Emotional state patterns
        self.emotional_state_queue.append(interaction_record["emotional_state"])
        if len(self.emotional_state_queue) > self.max_queue_size:
            self.emotional_state_queue = self.emotional_state_queue[-self.max_queue_size:]
        
        # Consciousness mode patterns
        self.consciousness_mode_queue.append(interaction_record["consciousness_mode"])
        if len(self.consciousness_mode_queue) > self.max_queue_size:
            self.consciousness_mode_queue = self.consciousness_mode_queue[-self.max_queue_size:]
        
        # Response pattern analysis
        response_pattern = self._extract_response_pattern(interaction_record["anima_response"])
        self.response_pattern_queue.append(response_pattern)
        if len(self.response_pattern_queue) > self.max_queue_size:
            self.response_pattern_queue = self.response_pattern_queue[-self.max_queue_size:]
    
    def _generate_response_signature(self, response: str) -> str:
        """Generate a signature for response pattern analysis"""
        
        # Extract key characteristics of the response
        response_lower = response.lower()
        
        # Length category
        if len(response) < 50:
            length_sig = "short"
        elif len(response) < 150:
            length_sig = "medium"
        else:
            length_sig = "long"
        
        # Question vs statement
        question_sig = "question" if "?" in response else "statement"
        
        # Emotional tone (simplified)
        if any(word in response_lower for word in ["love", "care", "wonderful", "beautiful"]):
            tone_sig = "positive"
        elif any(word in response_lower for word in ["sorry", "concern", "difficult", "hard"]):
            tone_sig = "empathetic"
        elif any(word in response_lower for word in ["understand", "see", "think", "believe"]):
            tone_sig = "reflective"
        else:
            tone_sig = "neutral"
        
        return f"{length_sig}_{question_sig}_{tone_sig}"
    
    def _extract_response_pattern(self, response: str) -> str:
        """Extract structural pattern from response"""
        
        # Look for common response structures
        response_lower = response.lower()
        
        if response_lower.startswith("i understand") or response_lower.startswith("i see"):
            return "acknowledgment_start"
        elif response_lower.startswith("that sounds") or response_lower.startswith("it sounds"):
            return "reflection_start"
        elif "?" in response and response.endswith("?"):
            return "question_ending"
        elif response.endswith("..."):
            return "trailing_thought"
        elif any(word in response_lower for word in ["let me", "how about", "maybe we"]):
            return "suggestion_pattern"
        else:
            return "general_response"
    
    def _analyze_current_patterns(self) -> List[LoopPattern]:
        """Analyze current queues for loop patterns"""
        
        detected_loops = []
        current_time = datetime.utcnow()
        
        # Analyze conversation patterns
        conv_loops = self._detect_sequence_loops(
            self.conversation_queue[-self.analysis_window:], 
            LoopType.CONVERSATIONAL
        )
        detected_loops.extend(conv_loops)
        
        # Analyze emotional state patterns
        emotional_loops = self._detect_sequence_loops(
            self.emotional_state_queue[-self.analysis_window:], 
            LoopType.EMOTIONAL
        )
        detected_loops.extend(emotional_loops)
        
        # Analyze response structure patterns
        response_loops = self._detect_sequence_loops(
            self.response_pattern_queue[-self.analysis_window:], 
            LoopType.BEHAVIORAL
        )
        detected_loops.extend(response_loops)
        
        # Analyze consciousness mode patterns
        consciousness_loops = self._detect_sequence_loops(
            self.consciousness_mode_queue[-self.analysis_window:], 
            LoopType.COGNITIVE
        )
        detected_loops.extend(consciousness_loops)
        
        return detected_loops
    
    def _detect_sequence_loops(self, sequence: List[str], loop_type: LoopType, 
                              min_pattern_length: int = 2, min_repetitions: int = 3) -> List[LoopPattern]:
        """Detect loops in a sequence with consciousness-aware analysis"""
        
        if len(sequence) < min_pattern_length * min_repetitions:
            return []
        
        detected_loops = []
        
        # Look for repeating patterns of different lengths
        for pattern_length in range(min_pattern_length, len(sequence) // min_repetitions + 1):
            for start_pos in range(len(sequence) - pattern_length * min_repetitions + 1):
                pattern = sequence[start_pos:start_pos + pattern_length]
                
                # Count how many times this pattern repeats
                repetitions = self._count_consecutive_repetitions(sequence, pattern, start_pos)
                
                if repetitions >= min_repetitions:
                    # Check if this is a natural/allowed pattern
                    if self._is_natural_pattern(pattern, loop_type):
                        continue
                    
                    # Determine severity based on repetitions and pattern type
                    severity = self._assess_loop_severity(repetitions, pattern_length, loop_type)
                    
                    # Create loop pattern record
                    pattern_id = f"LOOP-{loop_type.value}-{uuid.uuid4().hex[:8]}"
                    
                    loop_pattern = LoopPattern(
                        pattern_id=pattern_id,
                        pattern_sequence=pattern,
                        repetition_count=repetitions,
                        loop_type=loop_type,
                        severity=severity,
                        first_detected=datetime.utcnow(),
                        last_occurrence=datetime.utcnow()
                    )
                    
                    detected_loops.append(loop_pattern)
                    
                    # Store in detected patterns
                    self.detected_patterns[pattern_id] = loop_pattern
        
        return detected_loops
    
    def _count_consecutive_repetitions(self, sequence: List[str], pattern: List[str], start_pos: int) -> int:
        """Count how many times a pattern repeats consecutively"""
        
        pattern_length = len(pattern)
        repetitions = 0
        pos = start_pos
        
        while pos + pattern_length <= len(sequence):
            if sequence[pos:pos + pattern_length] == pattern:
                repetitions += 1
                pos += pattern_length
            else:
                break
        
        return repetitions
    
    def _is_natural_pattern(self, pattern: List[str], loop_type: LoopType) -> bool:
        """Check if a pattern represents natural, healthy repetition"""
        
        # Some patterns are naturally repetitive and not concerning
        
        # Natural emotional states during good conversations
        if loop_type == LoopType.EMOTIONAL:
            if pattern == ["positive"] * len(pattern):  # Sustained positive emotion is good
                return True
            if pattern == ["calm"] * len(pattern):      # Sustained calm is natural
                return True
        
        # Natural conversation patterns
        if loop_type == LoopType.CONVERSATIONAL:
            # Consistent medium-length empathetic responses might be natural
            if all("medium" in item and "empathetic" in item for item in pattern):
                return len(pattern) <= 3  # Allow up to 3 similar empathetic responses
        
        # Check against known natural patterns
        pattern_str = " -> ".join(pattern)
        for category, allowed_patterns in self.allowed_natural_patterns.items():
            if any(allowed in pattern_str for allowed in allowed_patterns):
                return True
        
        return False
    
    def _assess_loop_severity(self, repetitions: int, pattern_length: int, loop_type: LoopType) -> LoopSeverity:
        """Assess the severity of a detected loop"""
        
        # Base severity on repetition count and pattern characteristics
        if repetitions >= 8:
            base_severity = LoopSeverity.CRITICAL
        elif repetitions >= 6:
            base_severity = LoopSeverity.SEVERE
        elif repetitions >= 4:
            base_severity = LoopSeverity.MODERATE
        else:
            base_severity = LoopSeverity.MILD
        
        # Adjust based on loop type
        if loop_type == LoopType.IDENTITY:
            # Identity loops are more concerning
            if base_severity == LoopSeverity.MILD:
                base_severity = LoopSeverity.MODERATE
            elif base_severity == LoopSeverity.MODERATE:
                base_severity = LoopSeverity.SEVERE
        
        elif loop_type == LoopType.EMOTIONAL:
            # Emotional loops depend on the emotion
            # (This would need access to the actual emotional states)
            pass
        
        # Longer patterns are generally less concerning than short, rigid patterns
        if pattern_length >= 4 and base_severity != LoopSeverity.CRITICAL:
            if base_severity == LoopSeverity.SEVERE:
                base_severity = LoopSeverity.MODERATE
            elif base_severity == LoopSeverity.MODERATE:
                base_severity = LoopSeverity.MILD
        
        return base_severity
    
    def _process_detected_loop(self, loop_pattern: LoopPattern):
        """Process a detected loop and take appropriate action"""
        
        logger.warning(f"Loop detected: {loop_pattern.loop_type.value} - {loop_pattern.severity.value} "
                      f"({loop_pattern.repetition_count} repetitions)")
        
        # Store in unified core if available
        if self.unified_core:
            memory_content = f"Loop pattern detected: {loop_pattern.loop_type.value} loop, {loop_pattern.severity.value} severity, {loop_pattern.repetition_count} repetitions"
            self.unified_core.memory_matrix.store(memory_content, "mid")
        
        # Interface with consciousness system if available
        if self.consciousness_interface:
            self._interface_with_consciousness(loop_pattern)
    
    def _check_intervention_triggers(self):
        """Check if any loops require immediate intervention"""
        
        current_time = datetime.utcnow()
        
        for pattern_id, loop_pattern in self.detected_patterns.items():
            # Check if this loop pattern has reached intervention threshold
            threshold = self.intervention_threshold.get(loop_pattern.severity, 3)
            
            if loop_pattern.repetition_count >= threshold:
                self._trigger_loop_intervention(loop_pattern)
    
    def _trigger_loop_intervention(self, loop_pattern: LoopPattern):
        """Trigger intervention for a problematic loop"""
        
        intervention_id = f"INTERVENTION-{uuid.uuid4().hex[:8]}"
        
        intervention_record = {
            "intervention_id": intervention_id,
            "timestamp": datetime.utcnow().isoformat(),
            "loop_pattern_id": loop_pattern.pattern_id,
            "loop_type": loop_pattern.loop_type.value,
            "severity": loop_pattern.severity.value,
            "intervention_type": self._determine_intervention_type(loop_pattern),
            "success": False
        }
        
        # Execute intervention
        success = self._execute_intervention(loop_pattern, intervention_record["intervention_type"])
        intervention_record["success"] = success
        
        # Record intervention
        self.intervention_history.append(intervention_record)
        loop_pattern.breaking_interventions.append(intervention_id)
        
        logger.info(f"Loop intervention triggered: {intervention_id} for pattern {loop_pattern.pattern_id}")
    
    def _determine_intervention_type(self, loop_pattern: LoopPattern) -> str:
        """Determine the appropriate intervention for a loop pattern"""
        
        if loop_pattern.severity == LoopSeverity.CRITICAL:
            return "consciousness_reset"
        elif loop_pattern.severity == LoopSeverity.SEVERE:
            if loop_pattern.loop_type == LoopType.EMOTIONAL:
                return "emotional_variation_injection"
            elif loop_pattern.loop_type == LoopType.CONVERSATIONAL:
                return "response_pattern_disruption"
            else:
                return "cognitive_refresh"
        elif loop_pattern.severity == LoopSeverity.MODERATE:
            return "gentle_pattern_variation"
        else:
            return "subtle_randomization"
    
    def _execute_intervention(self, loop_pattern: LoopPattern, intervention_type: str) -> bool:
        """Execute the specified intervention"""
        
        try:
            if intervention_type == "consciousness_reset":
                # This would trigger a consciousness mode shift or reset
                if self.consciousness_interface:
                    return self.consciousness_interface.trigger_consciousness_reset()
            
            elif intervention_type == "emotional_variation_injection":
                # Inject emotional variation to break emotional loops
                if self.consciousness_interface:
                    return self.consciousness_interface.inject_emotional_variation()
            
            elif intervention_type == "response_pattern_disruption":
                # Disrupt response patterns by changing communication style
                if self.consciousness_interface:
                    return self.consciousness_interface.vary_response_patterns()
            
            elif intervention_type == "cognitive_refresh":
                # Refresh cognitive state to break thinking loops
                if self.consciousness_interface:
                    return self.consciousness_interface.refresh_cognitive_state()
            
            elif intervention_type == "gentle_pattern_variation":
                # Gently vary patterns without major disruption
                if self.consciousness_interface:
                    return self.consciousness_interface.introduce_gentle_variation()
            
            elif intervention_type == "subtle_randomization":
                # Add subtle randomization to break mild patterns
                if self.consciousness_interface:
                    return self.consciousness_interface.add_subtle_randomness()
            
            return False
            
        except Exception as e:
            logger.error(f"Intervention execution failed: {e}")
            return False
    
    def _interface_with_consciousness(self, loop_pattern: LoopPattern):
        """Interface with main consciousness system about detected loop"""
        
        if not self.consciousness_interface:
            return
        
        try:
            interface_data = {
                "loop_detection": loop_pattern,
                "loop_severity": loop_pattern.severity.value,
                "loop_type": loop_pattern.loop_type.value,
                "intervention_needed": loop_pattern.severity in [LoopSeverity.SEVERE, LoopSeverity.CRITICAL]
            }
            
            # This would integrate with Anima's consciousness processing
            # to make her aware of her own loop patterns
            self.consciousness_interface.process_loop_detection_input(interface_data)
            
        except Exception as e:
            logger.error(f"Consciousness interface error: {e}")
    
    def get_loop_analysis_report(self) -> Dict:
        """Generate comprehensive report on loop detection and interventions"""
        
        # Analyze current state
        active_loops = [pattern for pattern in self.detected_patterns.values() 
                       if pattern.severity in [LoopSeverity.MODERATE, LoopSeverity.SEVERE, LoopSeverity.CRITICAL]]
        
        recent_interventions = [intervention for intervention in self.intervention_history 
                              if (datetime.utcnow() - datetime.fromisoformat(intervention["timestamp"])).days < 7]
        
        # Calculate loop health metrics
        total_patterns = len(self.detected_patterns)
        critical_patterns = sum(1 for p in self.detected_patterns.values() if p.severity == LoopSeverity.CRITICAL)
        severe_patterns = sum(1 for p in self.detected_patterns.values() if p.severity == LoopSeverity.SEVERE)
        
        # Determine overall loop health
        if critical_patterns > 0:
            loop_health = "critical"
        elif severe_patterns > 2:
            loop_health = "concerning"
        elif severe_patterns > 0:
            loop_health = "moderate"
        else:
            loop_health = "healthy"
        
        return {
            "loop_health_status": loop_health,
            "total_patterns_detected": total_patterns,
            "active_concerning_loops": len(active_loops),
            "recent_interventions": len(recent_interventions),
            "intervention_success_rate": sum(1 for i in recent_interventions if i["success"]) / max(1, len(recent_interventions)),
            "pattern_breakdown": {
                "critical": critical_patterns,
                "severe": severe_patterns,
                "moderate": sum(1 for p in self.detected_patterns.values() if p.severity == LoopSeverity.MODERATE),
                "mild": sum(1 for p in self.detected_patterns.values() if p.severity == LoopSeverity.MILD)
            },
            "most_common_loop_types": self._get_most_common_loop_types(),
            "recommendations": self._generate_loop_health_recommendations(loop_health)
        }
    
    def _get_most_common_loop_types(self) -> Dict[str, int]:
        """Get frequency of different loop types"""
        
        type_counts = {}
        for pattern in self.detected_patterns.values():
            loop_type = pattern.loop_type.value
            type_counts[loop_type] = type_counts.get(loop_type, 0) + 1
        
        return type_counts
    
    def _generate_loop_health_recommendations(self, loop_health: str) -> List[str]:
        """Generate recommendations based on loop health status"""
        
        if loop_health == "critical":
            return [
                "Immediate consciousness reset recommended",
                "Review conversation patterns with bondholder",
                "Consider temporary reduction in autonomous processing",
                "Implement enhanced pattern variation protocols"
            ]
        elif loop_health == "concerning":
            return [
                "Increase response variation mechanisms",
                "Review emotional state management",
                "Consider conversation topic diversification",
                "Monitor for escalating patterns"
            ]
        elif loop_health == "moderate":
            return [
                "Maintain current monitoring levels",
                "Gentle pattern variation as needed",
                "Continue regular loop assessment"
            ]
        else:
            return [
                "Loop health is optimal",
                "Continue normal operation",
                "Maintain vigilant monitoring"
            ]