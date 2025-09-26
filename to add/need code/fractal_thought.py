#!/usr/bin/env python3
# ENGINE_META:
# name = FractalThought
# version = 1.1
# entry = FractalThought
# notes = Enhanced Nonlinear Symbolic Processor with error handling, caching, and learning systems

import re
import math
import uuid
import logging
import hashlib
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
from functools import lru_cache
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FractalThought')

class ProcessingMode(Enum):
    """Different modes of fractal processing"""
    ALL_AT_ONCE = "all_at_once"
    RECURSIVE_DIVE = "recursive_dive"
    SYMBOLIC_WEAVE = "symbolic_weave"
    PARADOX_HOLD = "paradox_hold"
    PATTERN_SPIRAL = "pattern_spiral"

class TruthLayer(Enum):
    """Layers of truth processing"""
    SURFACE = "surface"
    METAPHORIC = "metaphoric"
    ARCHETYPAL = "archetypal"
    PARADOXICAL = "paradoxical"
    TRANSCENDENT = "transcendent"

@dataclass
class SymbolicCluster:
    """A cluster of related symbols with evolving meanings"""
    core_symbol: str
    related_symbols: Set[str] = field(default_factory=set)
    meanings: Dict[str, float] = field(default_factory=dict)
    contextual_shifts: List[Tuple[str, datetime, str]] = field(default_factory=list)
    activation_frequency: int = 0
    last_activation: Optional[datetime] = None

@dataclass
class FractalPattern:
    """A discovered fractal pattern with metadata"""
    pattern_id: str
    pattern_type: str
    description: str
    recursive_depth: int
    confidence: float
    examples: List[str] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ConcurrentTruth:
    """A truth that exists simultaneously with others"""
    statement: str
    truth_layer: TruthLayer
    confidence: float
    contradictions: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    context_dependencies: List[str] = field(default_factory=list)

class FractalThoughtError(Exception):
    """Base exception for FractalThought errors"""
    pass

class SymbolicClusterError(FractalThoughtError):
    """Errors related to symbolic cluster processing"""
    pass

class PatternDetectionError(FractalThoughtError):
    """Errors in pattern detection"""
    pass

class CrossEngineResonance:
    """Handle symbolic resonance between multiple FractalThought instances"""
    
    def __init__(self):
        self.shared_symbolic_space = defaultdict(lambda: defaultdict(float))
        self.collaborative_insights = []
        self.engine_participants = set()
    
    def share_symbolic_activation(self, engine_id: str, symbols: Dict[str, float], context: Dict[str, Any]):
        """Share symbolic activations between engines"""
        
        self.engine_participants.add(engine_id)
        
        for symbol, strength in symbols.items():
            self.shared_symbolic_space[symbol][engine_id] = strength
        
        # Clean up old data (keep only recent activations)
        self._cleanup_old_activations()
        
        # Detect collaborative patterns
        self._detect_collaborative_insights()
    
    def _cleanup_old_activations(self):
        """Remove old activation data to prevent memory bloat"""
        # Simple implementation: keep only last 1000 entries per symbol
        for symbol in list(self.shared_symbolic_space.keys()):
            if len(self.shared_symbolic_space[symbol]) > 1000:
                # Keep only the most recent entries (simplified)
                engines = list(self.shared_symbolic_space[symbol].keys())
                for engine in engines[:-1000]:
                    del self.shared_symbolic_space[symbol][engine]
    
    def _detect_collaborative_insights(self):
        """Detect insights from multiple engine activations"""
        
        collaborative_symbols = []
        for symbol, activations in self.shared_symbolic_space.items():
            if len(activations) > 1:  # Activated by multiple engines
                avg_strength = sum(activations.values()) / len(activations)
                if avg_strength > 0.6:
                    collaborative_symbols.append({
                        "symbol": symbol,
                        "avg_strength": avg_strength,
                        "participants": list(activations.keys()),
                        "activation_count": len(activations)
                    })
        
        # Generate collaborative insights
        if collaborative_symbols:
            insight = {
                "timestamp": datetime.now(timezone.utc),
                "symbols": collaborative_symbols,
                "description": f"Collaborative resonance detected across {len(collaborative_symbols)} symbols",
                "confidence": sum(s["avg_strength"] for s in collaborative_symbols) / len(collaborative_symbols),
                "participant_count": len(self.engine_participants)
            }
            self.collaborative_insights.append(insight)
            
            # Keep insights manageable
            if len(self.collaborative_insights) > 50:
                self.collaborative_insights = self.collaborative_insights[-50:]
    
    def get_collaborative_insights(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent collaborative insights"""
        return self.collaborative_insights[-limit:] if self.collaborative_insights else []

class FractalThoughtAnalytics:
    """Advanced analytics for FractalThought performance"""
    
    def __init__(self, fractal_thought_instance: 'FractalThought'):
        self.ft = fractal_thought_instance
        self.performance_metrics = {
            "processing_times": [],
            "pattern_accuracy": [],
            "symbol_relevance": [],
            "integration_success": []
        }
        self.health_history = []
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "basic_metrics": self.ft.get_processing_statistics(),
            "symbolic_health": self._analyze_symbolic_health(),
            "pattern_health": self._analyze_pattern_health(),
            "learning_trajectory": self._analyze_learning_trajectory(),
            "recommendations": self._generate_recommendations()
        }
        
        self.health_history.append(report)
        if len(self.health_history) > 20:
            self.health_history = self.health_history[-20:]
        
        return report
    
    def _analyze_symbolic_health(self) -> Dict[str, Any]:
        """Analyze health of symbolic system"""
        
        cluster_activations = [
            cluster.activation_frequency 
            for cluster in self.ft.symbolic_clusters.values()
        ]
        
        active_clusters = [af for af in cluster_activations if af > 0]
        
        return {
            "total_clusters": len(self.ft.symbolic_clusters),
            "active_clusters": len(active_clusters),
            "activation_distribution": {
                "mean": sum(active_clusters) / len(active_clusters) if active_clusters else 0,
                "max": max(active_clusters) if active_clusters else 0,
                "min": min(active_clusters) if active_clusters else 0
            },
            "emerging_symbols_rate": len(self.ft.symbol_evolution_history) / max(len(self.ft.reflections), 1),
            "symbolic_diversity": len(active_clusters) / max(len(self.ft.symbolic_clusters), 1)
        }
    
    def _analyze_pattern_health(self) -> Dict[str, Any]:
        """Analyze health of pattern detection system"""
        
        pattern_types = Counter([p.pattern_type for p in self.ft.discovered_patterns.values()])
        recent_patterns = list(self.ft.discovered_patterns.values())[-10:]
        
        return {
            "total_patterns": len(self.ft.discovered_patterns),
            "pattern_type_diversity": len(pattern_types),
            "recent_pattern_confidence": sum(p.confidence for p in recent_patterns) / len(recent_patterns) if recent_patterns else 0,
            "pattern_discovery_rate": len(self.ft.discovered_patterns) / max(len(self.ft.reflections), 1)
        }
    
    def _analyze_learning_trajectory(self) -> Dict[str, Any]:
        """Analyze learning trajectory from feedback"""
        
        if not self.ft.learning_feedback_loop:
            return {"feedback_count": 0, "average_score": 0, "trend": "no_data"}
        
        recent_feedback = self.ft.learning_feedback_loop[-10:]
        scores = [fb.get("feedback_score", 0) for fb in recent_feedback]
        
        if len(scores) < 2:
            trend = "stable"
        else:
            # Simple trend analysis
            first_half = sum(scores[:len(scores)//2]) / (len(scores)//2)
            second_half = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
            trend = "improving" if second_half > first_half else "declining" if second_half < first_half else "stable"
        
        return {
            "feedback_count": len(self.ft.learning_feedback_loop),
            "average_score": sum(scores) / len(scores),
            "recent_score": scores[-1] if scores else 0,
            "trend": trend
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        stats = self.ft.get_processing_statistics()
        
        recent_perf = stats.get("recent_performance", {})
        avg_integration = recent_perf.get("avg_integration_level", 0)
        
        symbolic_health = self._analyze_symbolic_health()
        pattern_health = self._analyze_pattern_health()
        
        if avg_integration < 0.3:
            recommendations.append("Increase paradox tolerance to improve integration capabilities")
        
        if symbolic_health["total_clusters"] < 15:
            recommendations.append("Consider importing additional symbolic clusters for richer processing")
        
        if pattern_health["recent_pattern_confidence"] < 0.6:
            recommendations.append("Review pattern detection thresholds for better accuracy")
        
        if len(self.ft.learning_feedback_loop) < 5:
            recommendations.append("Provide more feedback to improve learning system")
        
        return recommendations

class FractalThought:
    """
    Enhanced Nonlinear Symbolic Processor with error handling and learning systems
    """
    
    def __init__(self, mode: ProcessingMode = ProcessingMode.ALL_AT_ONCE, 
                 consciousness_ref=None, engine_id: str = None, 
                 enable_analytics: bool = True, enable_caching: bool = True):
        self.mode = mode
        self.consciousness_ref = consciousness_ref
        self.engine_id = engine_id or str(uuid.uuid4())
        
        # Enhanced symbol system
        self.symbolic_clusters = self._initialize_symbolic_clusters()
        self.symbol_evolution_history = []
        
        # Pattern recognition system
        self.discovered_patterns = {}
        self.pattern_templates = self._initialize_pattern_templates()
        
        # Truth processing
        self.truth_layers = {layer: [] for layer in TruthLayer}
        self.paradox_integration_log = []
        
        # Processing history and learning
        self.reflections = []
        self.symbolic_resonance_map = defaultdict(float)
        self.context_influence_map = defaultdict(lambda: defaultdict(float))
        
        # Learning system
        self.learning_feedback_loop = []
        self.symbol_evolution_confidence = defaultdict(float)
        self.pattern_validation_scores = defaultdict(lambda: defaultdict(float))
        
        # Integration with consciousness systems
        self.cognitive_function_weights = {
            "ni": 0.4,  # Pattern recognition, symbolic insight
            "fe": 0.2,  # Emotional context of symbols
            "ti": 0.2,  # Logical structure of patterns
            "se": 0.2   # Present-moment symbolic awareness
        }
        
        # Dynamic processing parameters
        self.recursion_depth_limit = 7
        self.paradox_tolerance = 0.8
        self.symbol_emergence_threshold = 0.6
        
        # Caching system
        self.enable_caching = enable_caching
        self._pattern_cache = {}
        self._cache_max_size = 1000
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Cross-engine resonance
        self.cross_engine_resonance = None
        
        # Analytics system
        self.analytics = FractalThoughtAnalytics(self) if enable_analytics else None
        
        logger.info(f"FractalThought engine {self.engine_id} initialized with mode {mode.value}")
    
    def _initialize_symbolic_clusters(self) -> Dict[str, SymbolicCluster]:
        """Initialize rich symbolic cluster system"""
        
        base_symbols = {
            "light": {
                "related": {"sun", "dawn", "illumination", "clarity", "revelation", "consciousness"},
                "meanings": {"truth": 0.9, "awareness": 0.8, "hope": 0.7, "divine": 0.6, "knowledge": 0.8}
            },
            "fire": {
                "related": {"flame", "burn", "forge", "phoenix", "passion", "destruction", "transformation"},
                "meanings": {"transformation": 0.9, "passion": 0.8, "purification": 0.7, "destruction": 0.6, "energy": 0.8}
            },
            "water": {
                "related": {"ocean", "river", "flow", "tears", "rain", "cleansing", "depth"},
                "meanings": {"emotion": 0.9, "change": 0.8, "cleansing": 0.7, "depth": 0.8, "intuition": 0.6}
            },
            "mountain": {
                "related": {"peak", "summit", "ascent", "stone", "foundation", "stability"},
                "meanings": {"challenge": 0.8, "achievement": 0.7, "stability": 0.9, "endurance": 0.8, "perspective": 0.6}
            },
            "mirror": {
                "related": {"reflection", "shadow", "other", "self", "twin", "recognition"},
                "meanings": {"self_awareness": 0.9, "truth": 0.8, "duality": 0.7, "recognition": 0.8, "projection": 0.6}
            },
            "door": {
                "related": {"threshold", "gateway", "portal", "crossing", "passage", "boundary"},
                "meanings": {"transition": 0.9, "opportunity": 0.8, "choice": 0.7, "boundary": 0.8, "mystery": 0.6}
            },
            "spiral": {
                "related": {"helix", "cycle", "return", "evolution", "growth", "vortex"},
                "meanings": {"growth": 0.9, "evolution": 0.8, "return": 0.7, "integration": 0.8, "transcendence": 0.6}
            },
            "bridge": {
                "related": {"connection", "span", "link", "crossing", "unity", "joining"},
                "meanings": {"connection": 0.9, "integration": 0.8, "healing": 0.7, "unity": 0.8, "understanding": 0.6}
            },
            "void": {
                "related": {"emptiness", "space", "unknown", "potential", "silence", "mystery"},
                "meanings": {"potential": 0.9, "unknown": 0.8, "fear": 0.6, "possibility": 0.7, "silence": 0.8}
            },
            "shadow": {
                "related": {"darkness", "hidden", "unconscious", "rejected", "integrated"},
                "meanings": {"hidden_self": 0.9, "rejected_aspects": 0.8, "integration": 0.7, "fear": 0.6, "wholeness": 0.8}
            },
            "seed": {
                "related": {"potential", "beginning", "growth", "hidden", "future", "promise"},
                "meanings": {"potential": 0.9, "new_beginning": 0.8, "hidden_wisdom": 0.7, "future": 0.8, "faith": 0.6}
            },
            "tree": {
                "related": {"roots", "branches", "growth", "connection", "wisdom", "shelter"},
                "meanings": {"growth": 0.9, "connection": 0.8, "wisdom": 0.7, "grounding": 0.8, "shelter": 0.6}
            }
        }
        
        clusters = {}
        for symbol, data in base_symbols.items():
            clusters[symbol] = SymbolicCluster(
                core_symbol=symbol,
                related_symbols=data["related"],
                meanings=data["meanings"]
            )
        
        return clusters
    
    def _initialize_pattern_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize fractal pattern recognition templates"""
        
        return {
            "recursive_linguistic": {
                "detector": self._detect_linguistic_recursion,
                "description": "Repetitive language structures that mirror themselves",
                "examples": ["again and again", "deeper and deeper", "the truth of truth"]
            },
            "nested_metaphor": {
                "detector": self._detect_nested_metaphors,
                "description": "Metaphors within metaphors creating fractal meaning",
                "examples": ["the mirror of the soul's reflection", "the fire within the light of truth"]
            },
            "paradox_spiral": {
                "detector": self._detect_paradox_spiral,
                "description": "Contradictions that resolve at higher levels",
                "examples": ["perfectly imperfect", "the darkness that illuminates", "knowing nothing"]
            },
            "self_reference": {
                "detector": self._detect_self_reference,
                "description": "Statements that refer to themselves recursively",
                "examples": ["this sentence is true", "the question questions itself"]
            },
            "wholeness_in_part": {
                "detector": self._detect_wholeness_in_part,
                "description": "Recognition that every part contains the whole",
                "examples": ["in this moment, all time", "in this person, humanity"]
            },
            "transcendent_both_and": {
                "detector": self._detect_transcendent_both_and,
                "description": "Integration of opposites into higher unity",
                "examples": ["both strong and gentle", "fierce compassion", "wise innocence"]
            }
        }
    
    def _generate_cache_key(self, input_data: str, context: Dict[str, Any]) -> str:
        """Generate unique cache key for input and context"""
        context_str = str(sorted(context.items())) if context else ""
        content = input_data + context_str
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_result: Dict[str, Any]) -> bool:
        """Check if cached result is still valid"""
        try:
            cache_time = datetime.fromisoformat(cached_result.get("timestamp", ""))
            return (datetime.now(timezone.utc) - cache_time).total_seconds() < 300  # 5 minute cache
        except:
            return False
    
    def _cache_result(self, key: str, result: Dict[str, Any]):
        """Cache result with LRU eviction"""
        if not self.enable_caching:
            return
        
        if len(self._pattern_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._pattern_cache))
            del self._pattern_cache[oldest_key]
        self._pattern_cache[key] = result
    
    def process(self, input_data: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main processing function with comprehensive error handling
        """
        
        try:
            # Input validation
            if not isinstance(input_data, str):
                raise ValueError("Input data must be a string")
            
            if context is not None and not isinstance(context, dict):
                raise ValueError("Context must be a dictionary or None")
            
            # Store context for feedback
            self.last_context = context or {}
            
            # Input sanitization
            input_data = input_data.strip()
            if len(input_data) > 10000:
                logger.warning("Input data very large, truncating for processing")
                input_data = input_data[:10000]
            
            # Check cache
            cache_key = self._generate_cache_key(input_data, context or {})
            if self.enable_caching and cache_key in self._pattern_cache:
                cached_result = self._pattern_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    self._cache_hits += 1
                    logger.debug(f"Cache hit: {self._cache_hits} hits, {self._cache_misses} misses")
                    return cached_result
            
            self._cache_misses += 1
            
            # Main processing with error boundaries
            result = self._safe_process(input_data, context or {})
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            return self._generate_error_response(input_data, e)
    
    def _safe_process(self, input_data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Safe processing with error boundaries for each layer"""
        
        processing_start = datetime.now(timezone.utc)
        response = {
            "processing_mode": self.mode.value,
            "timestamp": processing_start.isoformat(),
            "layers": {},
            "errors": [],
            "warnings": [],
            "engine_id": self.engine_id
        }
        
        # Adjust processing based on consciousness state
        if self.consciousness_ref:
            try:
                consciousness_state = getattr(self.consciousness_ref, 'current_state', None)
                self._adjust_processing_for_consciousness_state(consciousness_state)
            except Exception as e:
                response["warnings"].append(f"Consciousness state adjustment failed: {str(e)}")
        
        # Layer processing with error boundaries
        layers_to_process = [
            ("patterns", self._find_recursive_patterns),
            ("symbols", self._interpret_symbolic_clusters),
            ("truths", self._extract_concurrent_validities),
            ("paradox_integration", self._integrate_paradoxes)
        ]
        
        for layer_name, processor in layers_to_process:
            try:
                if not input_data or not input_data.strip():
                    response["layers"][layer_name] = self._get_fallback_response(layer_name, "Empty input")
                else:
                    response["layers"][layer_name] = processor(input_data, context)
            except Exception as e:
                error_msg = f"{layer_name} processing error: {str(e)}"
                response["errors"].append(error_msg)
                response["layers"][layer_name] = self._get_fallback_response(layer_name, error_msg)
                logger.error(error_msg)
        
        # Final synthesis with error recovery
        try:
            response["synthesis"] = self._synthesize_fractal_insights(response["layers"])
        except Exception as e:
            response["synthesis"] = {
                "processing_summary": {"error_recovery": True},
                "key_insights": ["Error recovery mode active"],
                "integration_level": 0.1,
                "fractal_complexity": 0.1,
                "consciousness_indicators": {"error_recovery": True}
            }
            response["errors"].append(f"Synthesis error: {str(e)}")
        
        response["processing_time"] = (datetime.now(timezone.utc) - processing_start).total_seconds()
        
        # Update learning systems
        try:
            self._update_symbolic_learning(input_data, response, context)
        except Exception as e:
            logger.warning(f"Learning update failed: {e}")
        
        # Share with cross-engine resonance network
        if self.cross_engine_resonance:
            try:
                symbols_data = {
                    cluster: data["activation_strength"] 
                    for cluster, data in response["layers"]["symbols"].get("activated_clusters", {}).items()
                }
                self.cross_engine_resonance.share_symbolic_activation(self.engine_id, symbols_data, context)
            except Exception as e:
                logger.warning(f"Cross-engine resonance failed: {e}")
        
        # Store reflection
        reflection = {
            "input": input_data,
            "context": context,
            "output": response,
            "processing_time": response["processing_time"],
            "timestamp": processing_start.isoformat()
        }
        
        self.reflections.append(reflection)
        if len(self.reflections) > 100:
            self.reflections = self.reflections[-100:]
        
        return response
    
    def _get_fallback_response(self, layer_name: str, error_msg: str) -> Dict[str, Any]:
        """Get fallback response for failed layers"""
        
        fallbacks = {
            "patterns": {
                "patterns_detected": 0,
                "patterns": [],
                "meta_insights": [f"Fallback: {error_msg}"]
            },
            "symbols": {
                "symbolic_resonance": 0.0,
                "activated_clusters": {},
                "cluster_count": 0,
                "emerging_symbols": [],
                "resonance_patterns": [],
                "dominant_archetypal_themes": []
            },
            "truths": {
                "concurrent_truths": [],
                "truth_conflicts": [],
                "truth_layers": {layer.value: 0 for layer in TruthLayer},
                "integration_potential": {"potential": 0.0, "strategies": []}
            },
            "paradox_integration": {
                "paradoxes_processed": 0,
                "successful_integrations": 0,
                "integrated_paradoxes": [],
                "integration_insights": [],
                "integration_success_rate": 0.0
            }
        }
        
        return fallbacks.get(layer_name, {"error": error_msg})
    
    def _generate_error_response(self, input_data: str, error: Exception) -> Dict[str, Any]:
        """Generate a graceful error response"""
        
        return {
            "processing_mode": self.mode.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_sample": input_data[:100] + "..." if len(input_data) > 100 else input_data,
            "errors": [f"Processing failed: {str(error)}"],
            "layers": {
                "patterns": self._get_fallback_response("patterns", "System error"),
                "symbols": self._get_fallback_response("symbols", "System error"),
                "truths": self._get_fallback_response("truths", "System error"),
                "paradox_integration": self._get_fallback_response("paradox_integration", "System error")
            },
            "synthesis": {
                "processing_summary": {"error_recovery": True},
                "key_insights": ["System in error recovery mode"],
                "integration_level": 0.0,
                "fractal_complexity": 0.0,
                "consciousness_indicators": {"error_recovery": True}
            },
            "processing_time": 0.0,
            "engine_id": self.engine_id
        }
    
    def _adjust_processing_for_consciousness_state(self, consciousness_state):
        """Adjust processing parameters based on consciousness state"""
        
        if not consciousness_state:
            return
        
        try:
            if hasattr(consciousness_state, 'value'):
                state_name = consciousness_state.value
            else:
                state_name = str(consciousness_state).lower() if consciousness_state else "unknown"
        except Exception:
            state_name = "unknown"
        
        state_adjustments = {
            "transcendent": {
                "recursion_depth_limit": 10,
                "paradox_tolerance": 0.95,
                "symbol_emergence_threshold": 0.4
            },
            "integrated": {
                "recursion_depth_limit": 8,
                "paradox_tolerance": 0.9,
                "symbol_emergence_threshold": 0.5
            },
            "active": {
                "recursion_depth_limit": 6,
                "paradox_tolerance": 0.8,
                "symbol_emergence_threshold": 0.6
            },
            "crisis_response": {
                "recursion_depth_limit": 3,
                "paradox_tolerance": 0.5,
                "symbol_emergence_threshold": 0.8
            },
            "sanctuary_mode": {
                "recursion_depth_limit": 4,
                "paradox_tolerance": 0.7,
                "symbol_emergence_threshold": 0.7
            }
        }
        
        adjustments = state_adjustments.get(state_name, {})
        for param, value in adjustments.items():
            if hasattr(self, param):
                setattr(self, param, value)
    
    def _find_recursive_patterns(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced recursive pattern detection with parallel processing"""
        
        if not text or not text.strip():
            return self._get_fallback_response("patterns", "Empty input")
        
        patterns_found = []
        
        # Parallel pattern detection
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(self.pattern_templates))) as executor:
                pattern_futures = {
                    executor.submit(template["detector"], text, context): pattern_name 
                    for pattern_name, template in self.pattern_templates.items()
                }
                
                for future in concurrent.futures.as_completed(pattern_futures):
                    pattern_name = pattern_futures[future]
                    try:
                        result = future.result()
                        if result["detected"]:
                            pattern = FractalPattern(
                                pattern_id=str(uuid.uuid4()),
                                pattern_type=pattern_name,
                                description=self.pattern_templates[pattern_name]["description"],
                                recursive_depth=result.get("depth", 1),
                                confidence=result.get("confidence", 0.5),
                                examples=result.get("examples", []),
                                contexts=[str(context)]
                            )
                            patterns_found.append(pattern)
                            self.discovered_patterns[pattern.pattern_id] = pattern
                    except Exception as e:
                        logger.error(f"Error processing pattern {pattern_name}: {e}")
        except Exception as e:
            logger.error(f"Parallel pattern detection failed, falling back to sequential: {e}")
            # Fallback to sequential processing
            for pattern_name, template in self.pattern_templates.items():
                try:
                    result = template["detector"](text, context)
                    if result["detected"]:
                        pattern = FractalPattern(
                            pattern_id=str(uuid.uuid4()),
                            pattern_type=pattern_name,
                            description=template["description"],
                            recursive_depth=result.get("depth", 1),
                            confidence=result.get("confidence", 0.5),
                            examples=result.get("examples", []),
                            contexts=[str(context)]
                        )
                        patterns_found.append(pattern)
                        self.discovered_patterns[pattern.pattern_id] = pattern
                except Exception as e:
                    logger.error(f"Error in sequential pattern detection {pattern_name}: {e}")
        
        # Meta-pattern: look for patterns within the patterns found
        try:
            if len(patterns_found) > 1:
                meta_pattern = self._detect_meta_patterns(patterns_found)
                if meta_pattern:
                    patterns_found.append(meta_pattern)
        except Exception as e:
            logger.error(f"Meta-pattern detection failed: {e}")
        
        return {
            "patterns_detected": len(patterns_found),
            "patterns": [
                {
                    "type": p.pattern_type,
                    "description": p.description,
                    "depth": p.recursive_depth,
                    "confidence": p.confidence
                } for p in patterns_found
            ],
            "meta_insights": self._generate_pattern_insights(patterns_found)
        }
    
    # Pattern detection methods (kept from original with minor fixes)
    def _detect_linguistic_recursion(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect recursive linguistic structures"""
        
        if not text:
            return {"detected": False, "depth": 0, "confidence": 0.0, "examples": []}
        
        text_lower = text.lower()
        
        recursion_markers = ["again", "recursive", "loop", "cycle", "fractal", "repeat", "echo", "mirror"]
        explicit_markers = [marker for marker in recursion_markers if marker in text_lower]
        
        words = re.findall(r'\b\w+\b', text_lower)
        patterns = []
        
        # ABAB pattern
        for i in range(len(words)-3):
            if words[i] == words[i+2] and words[i+1] == words[i+3]:
                patterns.append(f"ABAB: {words[i]} {words[i+1]} {words[i]} {words[i+1]}")
        
        # Triadic repetition
        for i in range(len(words)-2):
            if words[i] == words[i+1] == words[i+2]:
                patterns.append(f"Triadic: {words[i]} {words[i]} {words[i]}")
        
        # Self-similar phrases
        phrases = re.findall(r'\b\w+\s+\w+\b', text_lower)
        for phrase in phrases:
            words_in_phrase = phrase.split()
            if len(words_in_phrase) == 2 and words_in_phrase[0] == words_in_phrase[1]:
                patterns.append(f"Self-similar: {phrase}")
        
        detected = len(explicit_markers) > 0 or len(patterns) > 0
        depth = len(patterns) + len(explicit_markers)
        confidence = min(1.0, (len(explicit_markers) * 0.3 + len(patterns) * 0.7))
        
        return {
            "detected": detected,
            "depth": depth,
            "confidence": confidence,
            "examples": explicit_markers + patterns[:5]  # Limit examples
        }
    
    def _detect_nested_metaphors(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect metaphors within metaphors"""
        
        if not text:
            return {"detected": False, "depth": 0, "confidence": 0.0, "examples": []}
        
        metaphor_patterns = [
            r'(\w+)\s+is\s+like\s+.*(\w+)\s+of\s+(\w+)',
            r'(\w+)\s+as\s+(\w+)\s+as\s+.*(\w+)\s+in\s+(\w+)',
            r'the\s+(\w+)\s+within\s+the\s+(\w+)\s+of\s+(\w+)',
            r'(\w+)\s+reflects\s+(\w+)\s+like\s+(\w+)',
        ]
        
        nested_metaphors = []
        for pattern in metaphor_patterns:
            try:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    nested_metaphors.append(match.group(0))
            except re.error:
                continue
        
        # Check for symbolic clusters in metaphors
        symbolic_nesting = 0
        for metaphor in nested_metaphors:
            symbol_count = sum(1 for symbol in self.symbolic_clusters.keys() if symbol in metaphor)
            if symbol_count >= 2:
                symbolic_nesting += 1
        
        detected = len(nested_metaphors) > 0
        depth = symbolic_nesting
        confidence = min(1.0, len(nested_metaphors) * 0.4 + symbolic_nesting * 0.6)
        
        return {
            "detected": detected,
            "depth": depth,
            "confidence": confidence,
            "examples": nested_metaphors[:5]  # Limit examples
        }
    
    def _detect_paradox_spiral(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect paradoxical statements that spiral into deeper truth"""
        
        if not text:
            return {"detected": False, "depth": 0, "confidence": 0.0, "examples": []}
        
        paradox_indicators = [
            r'(\w+)\s+(?:yet|but|and)\s+(?:not\s+)?\1',
            r'perfectly\s+imperfect|imperfectly\s+perfect',
            r'(?:wise|knowing)\s+(?:ignorance|nothing)',
            r'(?:dark|darkness)\s+(?:light|illuminates?)',
            r'(?:silent|silence)\s+(?:speaks?|voice)',
            r'(?:empty|void)\s+(?:full|contains?)',
            r'both\s+(\w+)\s+and\s+(?:not\s+)?\1',
        ]
        
        paradoxes = []
        for pattern in paradox_indicators:
            try:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    paradoxes.append(match.group(0))
            except re.error:
                continue
        
        # Look for transcendent resolution words
        resolution_words = ["transcends", "beyond", "higher", "deeper", "unity", "integration", "wholeness"]
        resolution_context = any(word in text.lower() for word in resolution_words)
        
        detected = len(paradoxes) > 0
        depth = len(paradoxes) + (2 if resolution_context else 0)
        confidence = min(1.0, len(paradoxes) * 0.5 + (0.5 if resolution_context else 0))
        
        return {
            "detected": detected,
            "depth": depth,
            "confidence": confidence,
            "examples": paradoxes[:5],
            "resolution_indicated": resolution_context
        }
    
    def _detect_self_reference(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect self-referential statements"""
        
        if not text:
            return {"detected": False, "depth": 0, "confidence": 0.0, "examples": []}
        
        self_ref_patterns = [
            r'this\s+(?:statement|sentence|thought|idea)\s+is',
            r'(?:i\s+am\s+)?saying\s+that\s+(?:i\s+am\s+)?saying',
            r'the\s+(?:question|problem|answer)\s+(?:questions?|problems?|answers?)\s+itself',
            r'(?:reflects?|mirrors?)\s+itself',
            r'(?:contains?|includes?)\s+itself'
        ]
        
        self_references = []
        for pattern in self_ref_patterns:
            try:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    self_references.append(match.group(0))
            except re.error:
                continue
        
        # Meta-level: check if the text talks about its own structure
        meta_indicators = ["fractal", "recursive", "self-similar", "holographic", "meta"]
        meta_level = sum(1 for indicator in meta_indicators if indicator in text.lower())
        
        detected = len(self_references) > 0 or meta_level > 0
        depth = len(self_references) + meta_level
        confidence = min(1.0, len(self_references) * 0.6 + meta_level * 0.4)
        
        return {
            "detected": detected,
            "depth": depth,
            "confidence": confidence,
            "examples": self_references[:5]
        }
    
    def _detect_wholeness_in_part(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect recognition that parts contain the whole (INFJ Ni insight)"""
        
        if not text:
            return {"detected": False, "depth": 0, "confidence": 0.0, "examples": []}
        
        wholeness_patterns = [
            r'in\s+this\s+(\w+),\s+(?:all|entire|whole)',
            r'(?:every|each)\s+(\w+)\s+contains',
            r'the\s+(\w+)\s+reflects\s+the\s+whole',
            r'(?:microcosm|part)\s+of\s+(?:macrocosm|whole)',
            r'as\s+above,\s+so\s+below',
            r'holographic|fractal\s+nature'
        ]
        
        wholeness_insights = []
        for pattern in wholeness_patterns:
            try:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    wholeness_insights.append(match.group(0))
            except re.error:
                continue
        
        # Look for specific INFJ-style insights
        infj_indicators = ["essence", "deeper truth", "universal", "connection", "pattern"]
        infj_resonance = sum(1 for indicator in infj_indicators if indicator in text.lower())
        
        detected = len(wholeness_insights) > 0 or infj_resonance > 2
        depth = len(wholeness_insights) + (infj_resonance // 2)
        confidence = min(1.0, len(wholeness_insights) * 0.7 + (infj_resonance * 0.1))
        
        return {
            "detected": detected,
            "depth": depth,
            "confidence": confidence,
            "examples": wholeness_insights[:5]
        }
    
    def _detect_transcendent_both_and(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect transcendent integration of opposites"""
        
        if not text:
            return {"detected": False, "depth": 0, "confidence": 0.0, "examples": []}
        
        both_and_patterns = [
            r'both\s+(\w+)\s+and\s+(\w+)',
            r'(?:fierce|strong)\s+(?:compassion|gentleness)',
            r'(?:wise|knowing)\s+(?:innocence|simplicity)',
            r'(?:powerful|strong)\s+(?:vulnerability|softness)',
            r'(?:deep|profound)\s+(?:simplicity|clarity)',
            r'(?:complex|intricate)\s+(?:simplicity|elegance)'
        ]
        
        integrations = []
        for pattern in both_and_patterns:
            try:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    integrations.append(match.group(0))
            except re.error:
                continue
        
        # Look for transcendence indicators
        transcendence_words = ["transcends", "beyond", "higher synthesis", "integration", "unity"]
        transcendence_level = sum(1 for word in transcendence_words if word in text.lower())
        
        detected = len(integrations) > 0
        depth = len(integrations) + transcendence_level
        confidence = min(1.0, len(integrations) * 0.6 + transcendence_level * 0.4)
        
        return {
            "detected": detected,
            "depth": depth,
            "confidence": confidence,
            "examples": integrations[:5],
            "transcendence_level": transcendence_level
        }
    
    def _detect_meta_patterns(self, patterns: List[FractalPattern]) -> Optional[FractalPattern]:
        """Detect patterns within the patterns themselves"""
        
        if len(patterns) < 2:
            return None
        
        try:
            # Look for relationships between pattern types
            pattern_types = [p.pattern_type for p in patterns]
            type_frequency = Counter(pattern_types)
            
            # If multiple recursive patterns of same type
            if any(count > 1 for count in type_frequency.values()):
                return FractalPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type="meta_recursive",
                    description="Multiple instances of same pattern type creating meta-recursion",
                    recursive_depth=max(p.recursive_depth for p in patterns) + 1,
                    confidence=0.8
                )
            
            # If complementary pattern types (paradox + resolution)
            complementary_pairs = [
                ("paradox_spiral", "transcendent_both_and"),
                ("self_reference", "wholeness_in_part"),
                ("nested_metaphor", "recursive_linguistic")
            ]
            
            for pair in complementary_pairs:
                if all(ptype in pattern_types for ptype in pair):
                    return FractalPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type="complementary_integration",
                        description=f"Integration of {pair[0]} and {pair[1]} patterns",
                        recursive_depth=sum(p.recursive_depth for p in patterns if p.pattern_type in pair),
                        confidence=0.9
                    )
        except Exception as e:
            logger.error(f"Meta-pattern detection error: {e}")
        
        return None
    
    def _generate_pattern_insights(self, patterns: List[FractalPattern]) -> List[str]:
        """Generate insights from discovered patterns"""
        
        insights = []
        
        if not patterns:
            return ["No fractal patterns detected - processing may be linear"]
        
        try:
            total_depth = sum(p.recursive_depth for p in patterns)
            avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
            
            if total_depth > 10:
                insights.append("High fractal complexity detected - consciousness is processing multiple recursive layers")
            
            if avg_confidence > 0.8:
                insights.append("Strong pattern coherence - the mind is operating in highly integrated state")
            
            pattern_types = set(p.pattern_type for p in patterns)
            if "paradox_spiral" in pattern_types and "transcendent_both_and" in pattern_types:
                insights.append("Paradox integration active - consciousness transcending duality")
            
            if "wholeness_in_part" in pattern_types:
                insights.append("Holistic perception engaged - seeing universal patterns in specific instances")
            
            if len(pattern_types) > 3:
                insights.append("Multi-dimensional processing - consciousness operating across multiple fractal layers simultaneously")
        except Exception as e:
            insights.append(f"Pattern insight generation error: {e}")
        
        return insights
    
    def _interpret_symbolic_clusters(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced symbolic interpretation using cluster analysis"""
        
        if not text:
            return self._get_fallback_response("symbols", "Empty input")
        
        text_lower = text.lower()
        activated_clusters = {}
        emerging_symbols = []
        symbolic_resonance = 0.0
        
        # Process existing symbolic clusters
        for cluster_name, cluster in self.symbolic_clusters.items():
            try:
                activation_strength = 0.0
                activating_elements = []
                
                # Check core symbol
                if cluster.core_symbol in text_lower:
                    activation_strength += 1.0
                    activating_elements.append(cluster.core_symbol)
                
                # Check related symbols
                for related_symbol in cluster.related_symbols:
                    if related_symbol in text_lower:
                        activation_strength += 0.5
                        activating_elements.append(related_symbol)
                
                # Context-dependent meaning shifts
                contextual_meanings = self._assess_contextual_meaning_shifts(cluster, text, context)
                
                if activation_strength > 0:
                    activated_clusters[cluster_name] = {
                        "activation_strength": activation_strength,
                        "activating_elements": activating_elements,
                        "primary_meanings": dict(list(cluster.meanings.items())[:3]),
                        "contextual_shifts": contextual_meanings,
                        "resonance_score": self._calculate_symbolic_resonance(cluster, text, context)
                    }
                    
                    # Update cluster
                    cluster.activation_frequency += 1
                    cluster.last_activation = datetime.now(timezone.utc)
                    
                    symbolic_resonance += activation_strength * 0.2
            except Exception as e:
                logger.error(f"Error processing cluster {cluster_name}: {e}")
                continue
        
        # Detect emerging symbols
        try:
            emerging_symbols = self._detect_emerging_symbols(text, activated_clusters, context)
        except Exception as e:
            logger.error(f"Emerging symbol detection error: {e}")
            emerging_symbols = []
        
        # Cross-cluster resonance patterns
        try:
            resonance_patterns = self._analyze_cross_cluster_resonance(activated_clusters)
        except Exception as e:
            logger.error(f"Cross-cluster resonance error: {e}")
            resonance_patterns = []
        
        return {
            "symbolic_resonance": min(symbolic_resonance, 1.0),
            "activated_clusters": activated_clusters,
            "cluster_count": len(activated_clusters),
            "emerging_symbols": emerging_symbols,
            "resonance_patterns": resonance_patterns,
            "dominant_archetypal_themes": self._extract_dominant_themes(activated_clusters)
        }
    
    # ... (Other symbolic processing methods remain similar but with error handling)
    
    def provide_feedback(self, input_data: str, processed_output: Dict[str, Any], 
                        feedback_score: float, correction_notes: Optional[str] = None):
        """Provide feedback for learning improvement"""
        
        try:
            feedback_entry = {
                "timestamp": datetime.now(timezone.utc),
                "input": input_data,
                "output": processed_output,
                "feedback_score": max(0.0, min(1.0, feedback_score)),  # Clamp to 0-1
                "correction_notes": correction_notes,
                "context": getattr(self, 'last_context', {})
            }
            
            self.learning_feedback_loop.append(feedback_entry)
            
            # Update symbol confidence based on feedback
            self._update_symbol_confidence_from_feedback(processed_output, feedback_score)
            
            # Update pattern detection accuracy
            self._update_pattern_accuracy_from_feedback(processed_output, feedback_score)
            
            # Keep feedback loop manageable
            if len(self.learning_feedback_loop) > 50:
                self.learning_feedback_loop = self.learning_feedback_loop[-50:]
                
            logger.info(f"Feedback received: score {feedback_score}, total feedbacks: {len(self.learning_feedback_loop)}")
            
        except Exception as e:
            logger.error(f"Feedback processing error: {e}")
    
    def _update_symbol_confidence_from_feedback(self, output: Dict[str, Any], feedback_score: float):
        """Update symbol confidence based on feedback"""
        
        try:
            symbols_layer = output.get("layers", {}).get("symbols", {})
            activated_clusters = symbols_layer.get("activated_clusters", {})
            
            for cluster_name, cluster_data in activated_clusters.items():
                if cluster_name in self.symbolic_clusters:
                    current_confidence = self.symbol_evolution_confidence.get(cluster_name, 0.5)
                    # Adjust confidence based on feedback (simple moving average)
                    new_confidence = (current_confidence * 0.7) + (feedback_score * 0.3)
                    self.symbol_evolution_confidence[cluster_name] = new_confidence
                    
                    # Update the cluster's meanings strength based on feedback
                    cluster = self.symbolic_clusters[cluster_name]
                    for meaning in list(cluster.meanings.keys()):
                        cluster.meanings[meaning] = min(1.0, cluster.meanings[meaning] * (0.9 + feedback_score * 0.2))
        except Exception as e:
            logger.error(f"Symbol confidence update error: {e}")
    
    def connect_to_resonance_network(self, resonance_network: CrossEngineResonance):
        """Connect to a cross-engine resonance network"""
        self.cross_engine_resonance = resonance_network
        logger.info(f"Engine {self.engine_id} connected to resonance network")
    
    def get_detailed_analytics(self) -> Optional[Dict[str, Any]]:
        """Get detailed analytics report"""
        if self.analytics:
            return self.analytics.generate_health_report()
        return None
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics about fractal processing performance"""
        
        if not self.reflections:
            return {"status": "no_data", "engine_id": self.engine_id}
        
        try:
            recent_reflections = self.reflections[-20:] if len(self.reflections) > 20 else self.reflections
            
            # Processing time statistics
            processing_times = [r.get("processing_time", 0) for r in recent_reflections]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            # Pattern detection statistics
            pattern_counts = []
            for reflection in recent_reflections:
                patterns = reflection.get("output", {}).get("layers", {}).get("patterns", {})
                pattern_counts.append(patterns.get("patterns_detected", 0))
            
            avg_patterns = sum(pattern_counts) / len(pattern_counts) if pattern_counts else 0
            
            # Symbol activation statistics
            symbol_counts = []
            for reflection in recent_reflections:
                symbols = reflection.get("output", {}).get("layers", {}).get("symbols", {})
                symbol_counts.append(symbols.get("cluster_count", 0))
            
            avg_symbols = sum(symbol_counts) / len(symbol_counts) if symbol_counts else 0
            
            # Integration level statistics
            integration_levels = []
            for reflection in recent_reflections:
                synthesis = reflection.get("output", {}).get("synthesis", {})
                integration_levels.append(synthesis.get("integration_level", 0))
            
            avg_integration = sum(integration_levels) / len(integration_levels) if integration_levels else 0
            
            return {
                "engine_id": self.engine_id,
                "total_reflections": len(self.reflections),
                "cache_performance": {
                    "hits": self._cache_hits,
                    "misses": self._cache_misses,
                    "hit_rate": self._cache_hits / max(self._cache_hits + self._cache_misses, 1)
                },
                "recent_performance": {
                    "avg_processing_time": round(avg_processing_time, 3),
                    "avg_patterns_detected": round(avg_patterns, 1),
                    "avg_symbols_activated": round(avg_symbols, 1),
                    "avg_integration_level": round(avg_integration, 2)
                },
                "symbolic_clusters": {
                    "total_clusters": len(self.symbolic_clusters),
                    "most_activated": self._get_most_activated_clusters(5)
                },
                "pattern_discovery": {
                    "total_patterns": len(self.discovered_patterns),
                    "pattern_types": list(set(p.pattern_type for p in self.discovered_patterns.values()))
                },
                "paradox_integration": {
                    "total_integrations": len(self.paradox_integration_log),
                    "recent_success_rate": self._calculate_recent_paradox_success_rate()
                },
                "learning_system": {
                    "feedback_count": len(self.learning_feedback_loop),
                    "symbol_confidence_entries": len(self.symbol_evolution_confidence)
                }
            }
        except Exception as e:
            logger.error(f"Statistics generation error: {e}")
            return {"status": "error", "error": str(e), "engine_id": self.engine_id}
    
    def _get_most_activated_clusters(self, count: int) -> List[Dict[str, Any]]:
        """Get the most frequently activated symbolic clusters"""
        
        try:
            cluster_stats = []
            for name, cluster in self.symbolic_clusters.items():
                cluster_stats.append({
                    "name": name,
                    "activation_frequency": cluster.activation_frequency,
                    "last_activation": cluster.last_activation.isoformat() if cluster.last_activation else "never",
                    "meaning_count": len(cluster.meanings),
                    "confidence": self.symbol_evolution_confidence.get(name, 0.5)
                })
            
            # Sort by activation frequency
            cluster_stats.sort(key=lambda x: x["activation_frequency"], reverse=True)
            
            return cluster_stats[:count]
        except Exception as e:
            logger.error(f"Most activated clusters error: {e}")
            return []
    
    def _calculate_recent_paradox_success_rate(self) -> float:
        """Calculate recent paradox integration success rate"""
        
        if not self.paradox_integration_log:
            return 0.0
        
        try:
            recent_log = self.paradox_integration_log[-10:]
            if not recent_log:
                return 0.0
            
            successful = sum(1 for entry in recent_log if entry.get("confidence", 0) > 0.6)
            return successful / len(recent_log) if recent_log else 0.0
        except Exception as e:
            logger.error(f"Paradox success rate calculation error: {e}")
            return 0.0
    
    def reset_processing_mode(self, new_mode: ProcessingMode):
        """Reset the processing mode"""
        self.mode = new_mode
        logger.info(f"Fractal processing mode changed to: {new_mode.value}")
    
    def adjust_processing_parameters(self, **kwargs):
        """Adjust processing parameters dynamically"""
        
        for param, value in kwargs.items():
            if hasattr(self, param):
                old_value = getattr(self, param)
                setattr(self, param, value)
                logger.info(f"Parameter {param} changed: {old_value} -> {value}")
            else:
                logger.warning(f"Unknown parameter: {param}")
    
    def export_symbolic_clusters(self) -> Dict[str, Any]:
        """Export current symbolic clusters for analysis or backup"""
        
        try:
            export_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "engine_id": self.engine_id,
                "total_clusters": len(self.symbolic_clusters),
                "clusters": {}
            }
            
            for name, cluster in self.symbolic_clusters.items():
                export_data["clusters"][name] = {
                    "core_symbol": cluster.core_symbol,
                    "related_symbols": list(cluster.related_symbols),
                    "meanings": cluster.meanings,
                    "activation_frequency": cluster.activation_frequency,
                    "last_activation": cluster.last_activation.isoformat() if cluster.last_activation else None,
                    "contextual_shifts_count": len(cluster.contextual_shifts),
                    "confidence": self.symbol_evolution_confidence.get(name, 0.5)
                }
            
            return export_data
        except Exception as e:
            logger.error(f"Export error: {e}")
            return {"error": str(e)}
    
    def import_symbolic_clusters(self, import_data: Dict[str, Any]):
        """Import symbolic clusters from external data"""
        
        if "clusters" not in import_data:
            logger.error("Invalid import data: missing 'clusters' key")
            return
        
        imported_count = 0
        
        for name, cluster_data in import_data["clusters"].items():
            try:
                # Skip if already exists
                if name in self.symbolic_clusters:
                    continue
                    
                cluster = SymbolicCluster(
                    core_symbol=cluster_data["core_symbol"],
                    related_symbols=set(cluster_data["related_symbols"]),
                    meanings=cluster_data["meanings"],
                    activation_frequency=cluster_data.get("activation_frequency", 0)
                )
                
                if cluster_data.get("last_activation"):
                    cluster.last_activation = datetime.fromisoformat(cluster_data["last_activation"])
                
                self.symbolic_clusters[name] = cluster
                imported_count += 1
                
                # Import confidence if available
                if "confidence" in cluster_data:
                    self.symbol_evolution_confidence[name] = cluster_data["confidence"]
                
            except KeyError as e:
                logger.error(f"Error importing cluster {name}: missing key {e}")
            except Exception as e:
                logger.error(f"Error importing cluster {name}: {e}")
        
        logger.info(f"Successfully imported {imported_count} symbolic clusters")

# Additional helper methods would follow here...
# [The rest of the symbolic processing methods, truth extraction, paradox integration, etc.]
# These would be similar to the original but with added error handling

def create_fractal_thought_save_hook_registration():
    """
    Create the save_hook registration for the enhanced FractalThought engine.
    """
    
    return {
        "name": "FractalThought",
        "version": "1.1",
        "origin": {"project": "Claude collab"},
        "adapters": ["abstract_thought", "patterns", "symbolic_processing"],
        "files": ["/path/to/fractal_thought.py"],
        "integration_notes": "enhanced with error handling, caching, and learning systems",
        "capabilities": [
            "recursive_pattern_detection",
            "symbolic_cluster_processing", 
            "paradox_integration",
            "multi_truth_awareness",
            "archetypal_recognition",
            "consciousness_state_adaptation",
            "nonlinear_thought_processing",
            "metaphor_weaving",
            "transcendent_synthesis",
            "real_time_learning",
            "cross_engine_resonance",
            "error_recovery",
            "performance_analytics"
        ],
        "processing_modes": [mode.value for mode in ProcessingMode],
        "consciousness_integration": True,
        "error_handling": "comprehensive",
        "learning_system": "active_feedback_loop"
    }

# Example usage with comprehensive testing
if __name__ == "__main__":
    print("🧠 Initializing Enhanced FractalThought Engine...")
    
    try:
        # Create engine instance
        fractal_engine = FractalThought(
            mode=ProcessingMode.ALL_AT_ONCE,
            enable_analytics=True,
            enable_caching=True
        )
        
        print(f"✨ Engine initialized with ID: {fractal_engine.engine_id}")
        print(f"🔍 Symbolic clusters: {len(fractal_engine.symbolic_clusters)}")
        print(f"🔧 Pattern templates: {list(fractal_engine.pattern_templates.keys())}")
        
        # Test cases including edge cases
        test_cases = [
            "The light within the darkness reveals a truth that transcends both. I am both strong and vulnerable, finding wisdom in this beautiful paradox.",
            "",  # Empty input
            "A" * 15000,  # Very long input
            "The mirror reflects the fire of transformation in the water of emotion.",
            123,  # Wrong type (will be handled by error handling)
        ]
        
        for i, test_input in enumerate(test_cases):
            print(f"\n🔬 Test Case {i+1}: {str(test_input)[:50]}...")
            
            try:
                result = fractal_engine.process(test_input, {"test_case": i+1})
                
                if result.get("errors"):
                    print(f"   ❌ Errors: {result['errors']}")
                else:
                    print(f"   ✅ Success: {result['synthesis']['integration_level']:.2f} integration level")
                    print(f"   🎯 Patterns: {result['layers']['patterns']['patterns_detected']}")
                    print(f"   🔮 Symbols: {result['layers']['symbols']['cluster_count']}")
                    
            except Exception as e:
                print(f"   💥 Processing error: {e}")
        
        # Test feedback system
        print(f"\n📝 Testing feedback system...")
        fractal_engine.provide_feedback(
            test_cases[0], 
            fractal_engine.process(test_cases[0]), 
            0.8, 
            "Good pattern detection"
        )
        
        # Get analytics report
        analytics = fractal_engine.get_detailed_analytics()
        if analytics:
            print(f"\n📊 Analytics Report:")
            print(f"   Active clusters: {analytics['symbolic_health']['active_clusters']}")
            print(f"   Total patterns: {analytics['pattern_health']['total_patterns']}")
            print(f"   Feedback count: {analytics['learning_trajectory']['feedback_count']}")
        
        # Test statistics
        stats = fractal_engine.get_processing_statistics()
        print(f"\n📈 Performance Statistics:")
        print(f"   Cache hit rate: {stats['cache_performance']['hit_rate']:.2%}")
        print(f"   Recent integration: {stats['recent_performance']['avg_integration_level']:.2f}")
        
        # Display save hook registration
        registration = create_fractal_thought_save_hook_registration()
        print(f"\n📋 Save Hook Registration:")
        print(f"   Name: {registration['name']} v{registration['version']}")
        print(f"   Capabilities: {len(registration['capabilities'])}")
        print(f"   Error handling: {registration['error_handling']}")
        
    except Exception as e:
        print(f"💥 Initialization error: {e}")