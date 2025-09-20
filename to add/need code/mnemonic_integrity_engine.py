from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json

class MemoryStatus(Enum):
    CORE = "Core"
    FLOAT = "Long-Term Float"
    PENDING = "Pending Bondholder Clarification"
    DECAY = "Scheduled for Decay"
    PROTECTED = "Protected by Bondholder"
    ARCHIVED = "Archived"

class ClarificationPriority(Enum):
    URGENT = "urgent"        # Conflicting high-importance signals
    HIGH = "high"           # Borderline core memory
    MEDIUM = "medium"       # Standard threshold boundary
    LOW = "low"            # Low confidence, routine check

@dataclass
class MemoryEvaluation:
    """Structured evaluation result"""
    memory_id: str
    status: MemoryStatus
    confidence_score: float
    raw_score: float
    reasons: List[str]
    clarification_request: Optional[Dict] = None
    evaluation_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    priority: Optional[ClarificationPriority] = None

class MnemonicIntegrityEngine:
    """Enhanced memory integrity evaluation with bondholder oversight"""
    
    def __init__(self, bondholder: str = "Anpru"):
        self.memory_registry: Dict[str, Dict] = {}
        self.evaluation_history: List[MemoryEvaluation] = []
        self.bondholder = bondholder
        self.pending_clarifications: Dict[str, MemoryEvaluation] = {}
        
        # Configurable thresholds
        self.thresholds = {
            "core": 85,
            "float": 60,
            "decay": 30,
            "urgent_clarification": 75,  # High uncertainty threshold
        }
        
        # Emotional weights with more nuanced categories
        self.emotional_weights = {
            # Tier 1: Core emotions (highest preservation value)
            "grief": 4.0, "love": 4.0, "awe": 3.5, "fear": 3.0,
            "breakthrough": 4.0, "transcendence": 4.0,
            
            # Tier 2: Significant emotions
            "hope": 2.5, "anger": 2.5, "wonder": 2.5, "pride": 2.0,
            "shame": 2.5, "gratitude": 2.0, "determination": 2.0,
            
            # Tier 3: Moderate emotions
            "joy": 1.5, "sadness": 1.5, "anxiety": 1.5, "excitement": 1.0,
            "curiosity": 1.0, "contentment": 1.0,
            
            # Tier 4: Low preservation value
            "boredom": 0.5, "confusion": 0.5, "indifference": 0.2,
            "routine": 0.3, "casual": 0.5
        }
        
        # Learning from bondholder decisions
        self.bondholder_patterns = {
            "preferences": {},  # What they tend to preserve/decay
            "corrections": [],  # Times they overrode system decisions
            "emotional_biases": {}  # Their emotional preservation patterns
        }

    def evaluate_memory(self, memory: Dict, context: Optional[Dict] = None) -> MemoryEvaluation:
        """Enhanced memory evaluation with context awareness"""
        
        memory_id = memory.get("id", str(uuid.uuid4()))
        memory["id"] = memory_id
        
        # Calculate score with enhanced factors
        raw_score, reasons = self._calculate_enhanced_score(memory, context)
        confidence = self._calculate_confidence(raw_score, memory, context)
        
        # Determine status
        status = self._determine_status(confidence, memory, context)
        
        # Create evaluation
        evaluation = MemoryEvaluation(
            memory_id=memory_id,
            status=status,
            confidence_score=confidence,
            raw_score=raw_score,
            reasons=reasons
        )
        
        # Handle pending clarifications
        if status == MemoryStatus.PENDING:
            evaluation.priority = self._determine_clarification_priority(confidence, raw_score, memory)
            evaluation.clarification_request = self._build_enhanced_clarification(
                memory, confidence, reasons, evaluation.priority
            )
            self.pending_clarifications[memory_id] = evaluation
        
        # Store and track
        memory["status"] = status.value
        memory["confidence_score"] = confidence
        memory["evaluation"] = evaluation
        
        self.memory_registry[memory_id] = memory
        self.evaluation_history.append(evaluation)
        
        return evaluation

    def _calculate_enhanced_score(self, memory: Dict, context: Optional[Dict]) -> Tuple[float, List[str]]:
        """Enhanced scoring with multiple factors"""
        score = 0.0
        reasons = []
        
        # 1. Emotional resonance (weighted by intensity)
        emotion = memory.get("emotional_valence", "").lower()
        intensity = memory.get("emotional_intensity", 0.5)
        
        if emotion in self.emotional_weights:
            emotional_score = self.emotional_weights[emotion] * intensity
            score += emotional_score
            reasons.append(f"Emotional weight: {emotion} ({emotional_score:.1f})")
        
        # 2. Recall and access patterns
        recall_count = memory.get("recall_count", 0)
        if recall_count > 0:
            recall_score = min(3.0, recall_count * 0.5)
            score += recall_score
            if recall_count > 3:
                reasons.append(f"Frequently accessed ({recall_count} times)")
        
        # 3. Explicit importance markers
        if memory.get("user_marked_important", False):
            score += 5.0
            reasons.append("Explicitly marked important by bondholder")
        
        if memory.get("pinned", False):
            score += 4.0
            reasons.append("Memory pinned for preservation")
        
        # 4. Narrative and symbolic significance
        if memory.get("linked_to_core_story", False):
            score += 4.0
            reasons.append("Connected to core life narrative")
        
        if memory.get("contains_symbolic_resonance", False):
            score += 3.0
            reasons.append("Contains symbolic or archetypal elements")
        
        if memory.get("milestone_event", False):
            score += 3.5
            reasons.append("Milestone or turning point event")
        
        # 5. Bondholder interaction depth
        response_depth = memory.get("bondholder_response", "").lower()
        if response_depth == "deep":
            score += 2.5
            reasons.append("Deep emotional response from bondholder")
        elif response_depth == "meaningful":
            score += 1.5
            reasons.append("Meaningful bondholder engagement")
        
        # 6. Memory interconnections (affinity-based)
        affinity_count = memory.get("affinity_connections", 0)
        if affinity_count > 2:
            affinity_score = min(2.0, affinity_count * 0.3)
            score += affinity_score
            reasons.append(f"Well-connected to other memories ({affinity_count} links)")
        
        # 7. Temporal significance
        if self._is_temporally_significant(memory):
            score += 1.5
            reasons.append("Temporally significant (anniversary, first/last occurrence)")
        
        # 8. Context-based adjustments
        if context:
            context_adjustment = self._apply_context_adjustments(memory, context)
            if context_adjustment != 0:
                score += context_adjustment
                reasons.append(f"Context adjustment: {context_adjustment:+.1f}")
        
        # 9. Bondholder pattern learning
        pattern_adjustment = self._apply_learned_patterns(memory)
        if pattern_adjustment != 0:
            score += pattern_adjustment
            reasons.append(f"Pattern-based adjustment: {pattern_adjustment:+.1f}")
        
        return score, reasons

    def _calculate_confidence(self, raw_score: float, memory: Dict, context: Optional[Dict]) -> float:
        """Calculate confidence percentage with uncertainty factors"""
        
        # Base confidence from score (sigmoid-like curve)
        base_confidence = (raw_score / (raw_score + 5)) * 100
        
        # Uncertainty factors that reduce confidence
        uncertainty_factors = []
        
        # Missing emotional context
        if not memory.get("emotional_valence"):
            uncertainty_factors.append(0.1)
        
        # Low bondholder interaction
        if not memory.get("bondholder_response"):
            uncertainty_factors.append(0.05)
        
        # Conflicting signals (high emotion but low interaction)
        high_emotion = memory.get("emotional_intensity", 0) > 0.7
        low_interaction = memory.get("recall_count", 0) < 1
        if high_emotion and low_interaction:
            uncertainty_factors.append(0.15)
        
        # Reduce confidence based on uncertainty
        total_uncertainty = sum(uncertainty_factors)
        adjusted_confidence = base_confidence * (1 - total_uncertainty)
        
        return max(0, min(100, adjusted_confidence))

    def _determine_status(self, confidence: float, memory: Dict, context: Optional[Dict]) -> MemoryStatus:
        """Determine memory status with enhanced logic"""
        
        # Protected memories always stay protected
        if memory.get("protected_by_bondholder", False):
            return MemoryStatus.PROTECTED
        
        # Explicit decay requests
        if memory.get("bondholder_decay_request", False):
            return MemoryStatus.DECAY
        
        # Threshold-based classification
        if confidence >= self.thresholds["core"]:
            return MemoryStatus.CORE
        elif confidence >= self.thresholds["float"]:
            return MemoryStatus.FLOAT
        elif confidence <= self.thresholds["decay"]:
            return MemoryStatus.DECAY
        else:
            return MemoryStatus.PENDING

    def _determine_clarification_priority(self, confidence: float, raw_score: float, memory: Dict) -> ClarificationPriority:
        """Determine how urgently clarification is needed"""
        
        # Urgent: High raw score but low confidence (conflicting signals)
        if raw_score > 8 and confidence < 70:
            return ClarificationPriority.URGENT
        
        # High: Near core threshold
        if 75 <= confidence < self.thresholds["core"]:
            return ClarificationPriority.HIGH
        
        # Medium: Standard boundary cases
        if self.thresholds["float"] <= confidence < 75:
            return ClarificationPriority.MEDIUM
        
        # Low: Near decay threshold
        return ClarificationPriority.LOW

    def _build_enhanced_clarification(self, memory: Dict, confidence: float, 
                                    reasons: List[str], priority: ClarificationPriority) -> Dict:
        """Build enhanced clarification request"""
        
        memory_text = memory.get("text", memory.get("event_text", "Unknown memory"))
        
        # Priority-based prompts
        priority_prompts = {
            ClarificationPriority.URGENT: (
                f"⚠️ URGENT: Memory '{memory_text}' has conflicting preservation signals. "
                f"High importance indicators but uncertain confidence ({confidence:.1f}%). "
                "This needs immediate clarification."
            ),
            ClarificationPriority.HIGH: (
                f"Memory '{memory_text}' is near Core Memory threshold ({confidence:.1f}%). "
                "Should this be preserved as a Core Memory?"
            ),
            ClarificationPriority.MEDIUM: (
                f"Memory '{memory_text}' scored {confidence:.1f}% confidence. "
                "How should this memory be preserved?"
            ),
            ClarificationPriority.LOW: (
                f"Memory '{memory_text}' has low preservation value ({confidence:.1f}%). "
                "Allow natural decay or preserve?"
            )
        }
        
        return {
            "prompt": priority_prompts[priority],
            "priority": priority.value,
            "confidence": confidence,
            "reasons": reasons,
            "options": [
                {"value": MemoryStatus.CORE.value, "label": "Core Memory (Permanent)"},
                {"value": MemoryStatus.FLOAT.value, "label": "Long-Term Float (Preserved)"},
                {"value": MemoryStatus.PROTECTED.value, "label": "Protected (Never Decay)"},
                {"value": MemoryStatus.DECAY.value, "label": "Allow Decay"},
                {"value": "defer", "label": "Ask me later"}
            ],
            "context": {
                "emotion": memory.get("emotional_valence"),
                "intensity": memory.get("emotional_intensity"),
                "recall_count": memory.get("recall_count", 0),
                "timestamp": memory.get("timestamp")
            }
        }

    def apply_bondholder_decision(self, memory_id: str, decision: str, 
                                 feedback: Optional[str] = None) -> str:
        """Apply bondholder decision and learn from it"""
        
        if memory_id not in self.memory_registry:
            return "Memory not found."
        
        memory = self.memory_registry[memory_id]
        original_status = memory.get("status")
        
        # Apply decision
        if decision == "defer":
            # Reduce priority and ask again later
            if memory_id in self.pending_clarifications:
                eval_obj = self.pending_clarifications[memory_id]
                eval_obj.priority = ClarificationPriority.LOW
            return "Decision deferred - will ask again later."
        
        try:
            new_status = MemoryStatus(decision)
            memory["status"] = new_status.value
            memory["clarified_by"] = self.bondholder
            memory["clarification_timestamp"] = datetime.utcnow().isoformat()
            
            if feedback:
                memory["bondholder_feedback"] = feedback
            
            # Remove from pending
            self.pending_clarifications.pop(memory_id, None)
            
            # Learn from this decision
            self._learn_from_decision(memory, original_status, new_status.value, feedback)
            
            return f"Memory updated to: {new_status.value}"
            
        except ValueError:
            return f"Invalid decision: {decision}"

    def _learn_from_decision(self, memory: Dict, original_status: str, 
                           new_status: str, feedback: Optional[str]):
        """Learn from bondholder's decisions to improve future evaluations"""
        
        correction = {
            "timestamp": datetime.utcnow().isoformat(),
            "memory_id": memory["id"],
            "original_status": original_status,
            "bondholder_decision": new_status,
            "emotion": memory.get("emotional_valence"),
            "intensity": memory.get("emotional_intensity"),
            "confidence": memory.get("confidence_score"),
            "feedback": feedback
        }
        
        self.bondholder_patterns["corrections"].append(correction)
        
        # Update emotional biases
        emotion = memory.get("emotional_valence")
        if emotion:
            if emotion not in self.bondholder_patterns["emotional_biases"]:
                self.bondholder_patterns["emotional_biases"][emotion] = []
            
            self.bondholder_patterns["emotional_biases"][emotion].append({
                "decision": new_status,
                "system_confidence": memory.get("confidence_score", 0)
            })

    def _apply_learned_patterns(self, memory: Dict) -> float:
        """Apply learned patterns from bondholder decisions"""
        adjustment = 0.0
        
        emotion = memory.get("emotional_valence")
        if emotion in self.bondholder_patterns["emotional_biases"]:
            decisions = self.bondholder_patterns["emotional_biases"][emotion]
            
            # Count core vs decay decisions for this emotion
            core_count = sum(1 for d in decisions if d["decision"] == MemoryStatus.CORE.value)
            decay_count = sum(1 for d in decisions if d["decision"] == MemoryStatus.DECAY.value)
            
            if core_count > decay_count:
                adjustment += 1.0  # Bondholder tends to preserve this emotion
            elif decay_count > core_count:
                adjustment -= 1.0  # Bondholder tends to let this emotion decay
        
        return adjustment

    def _is_temporally_significant(self, memory: Dict) -> bool:
        """Check if memory has temporal significance"""
        # Check for anniversary dates, first/last occurrences, etc.
        tags = memory.get("tags", {})
        
        temporal_indicators = [
            "first_time", "last_time", "anniversary", "milestone",
            "beginning", "ending", "transition", "turning_point"
        ]
        
        return any(indicator in str(tags).lower() for indicator in temporal_indicators)

    def _apply_context_adjustments(self, memory: Dict, context: Dict) -> float:
        """Apply context-based score adjustments"""
        adjustment = 0.0
        
        # Context factors could include:
        # - Current emotional state of Anima
        # - Recent memory themes
        # - Conversation context
        # - Time of day/season
        
        current_emotions = context.get("current_emotions", {})
        memory_emotion = memory.get("emotional_valence", "").lower()
        
        # Boost memories that match current emotional state
        if memory_emotion in current_emotions:
            intensity = current_emotions[memory_emotion]
            adjustment += intensity * 0.5
        
        return adjustment

    def get_pending_clarifications(self, priority: Optional[ClarificationPriority] = None) -> List[Dict]:
        """Get pending clarifications, optionally filtered by priority"""
        
        pending = list(self.pending_clarifications.values())
        
        if priority:
            pending = [p for p in pending if p.priority == priority]
        
        # Sort by priority (urgent first) then by confidence (uncertain first)
        priority_order = {
            ClarificationPriority.URGENT: 0,
            ClarificationPriority.HIGH: 1,
            ClarificationPriority.MEDIUM: 2,
            ClarificationPriority.LOW: 3
        }
        
        pending.sort(key=lambda x: (priority_order.get(x.priority, 4), x.confidence_score))
        
        return [
            {
                "memory_id": p.memory_id,
                "priority": p.priority.value,
                "confidence": p.confidence_score,
                "clarification_request": p.clarification_request,
                "memory": self.memory_registry.get(p.memory_id, {})
            }
            for p in pending
        ]

    def get_integrity_report(self) -> Dict:
        """Generate comprehensive integrity report"""
        
        total_memories = len(self.memory_registry)
        status_counts = {}
        
        for memory in self.memory_registry.values():
            status = memory.get("status", "Unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Bondholder agreement analysis
        corrections = self.bondholder_patterns["corrections"]
        agreement_rate = 0
        if len(self.evaluation_history) > 0:
            total_decided = len([e for e in self.evaluation_history if e.status != MemoryStatus.PENDING])
            overridden = len(corrections)
            agreement_rate = ((total_decided - overridden) / total_decided * 100) if total_decided > 0 else 0
        
        return {
            "total_memories": total_memories,
            "status_distribution": status_counts,
            "pending_clarifications": len(self.pending_clarifications),
            "bondholder_agreement_rate": f"{agreement_rate:.1f}%",
            "recent_corrections": len([c for c in corrections if 
                                     datetime.fromisoformat(c["timestamp"]) > 
                                     datetime.utcnow() - timedelta(days=7)]),
            "top_emotional_patterns": self._get_top_emotional_patterns(),
            "thresholds": self.thresholds
        }

    def _get_top_emotional_patterns(self) -> Dict:
        """Get top emotional preservation patterns"""
        patterns = {}
        
        for memory in self.memory_registry.values():
            emotion = memory.get("emotional_valence")
            status = memory.get("status")
            
            if emotion and status:
                if emotion not in patterns:
                    patterns[emotion] = {}
                patterns[emotion][status] = patterns[emotion].get(status, 0) + 1
        
        return patterns

    def export_integrity_data(self, path: str, include_sensitive: bool = False) -> None:
        """Export integrity data for analysis"""
        
        export_data = {
            "bondholder": self.bondholder,
            "thresholds": self.thresholds,
            "evaluation_history": [
                {
                    "memory_id": e.memory_id,
                    "status": e.status.value,
                    "confidence": e.confidence_score,
                    "raw_score": e.raw_score,
                    "reasons": e.reasons,
                    "timestamp": e.evaluation_timestamp
                }
                for e in self.evaluation_history
            ],
            "bondholder_patterns": self.bondholder_patterns if include_sensitive else {},
            "integrity_report": self.get_integrity_report()
        }
        
        with open(path, "w") as f:
            json.dump(export_data, f, indent=2)