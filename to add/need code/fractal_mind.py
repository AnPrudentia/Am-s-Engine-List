# engines/fractal_mind_engine.py
"""
FractalMindEngine
Wrapper around the FractalThought processor so Anima can "think like you".
Produces a ConsciousnessResponse for the orchestrator to consume.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Union
import uuid
import traceback
import logging

from core.io_models import ConsciousnessResponse, ProcessingDepth, ConsciousnessInput
from core.engine_base import EngineBase
from fractal.fractal_thought import FractalThought, ProcessingMode
from core.anima_promise import AnimaPromise

# Configure logger
logger = logging.getLogger('FractalMindEngine')


class FractalMindEngine(EngineBase):
    """
    Enhanced engine wrapper that exposes FractalThought to orchestrator.
    - Constructs a FractalThought instance with proper configuration
    - Handles errors gracefully with detailed logging
    - Returns a properly formatted ConsciousnessResponse object
    """

    name = "FractalMind"

    def __init__(self, profile: Optional[Dict[str, Any]] = None):
        super().__init__(profile=profile) if hasattr(super(), "__init__") else None
        self.profile = profile or {}
        self.logger = logger.getChild(self.name)
        
        # Initialize processor with configuration from profile
        processing_mode = self._get_processing_mode_from_profile()
        self.processor = FractalThought(
            mode=processing_mode,
            consciousness_ref=self.profile.get("consciousness_ref"),
            engine_id=f"fractal_mind_{uuid.uuid4().hex[:8]}",
            enable_analytics=True,
            enable_caching=True
        )
        
        # Set up cross-engine resonance if available
        self._setup_cross_engine_resonance()
        
        self.logger.info(f"Initialized with mode: {processing_mode.value}")

    def _get_processing_mode_from_profile(self) -> ProcessingMode:
        """Get processing mode from profile with fallbacks"""
        mode_mapping = {
            "all_at_once": ProcessingMode.ALL_AT_ONCE,
            "recursive_dive": ProcessingMode.RECURSIVE_DIVE,
            "symbolic_weave": ProcessingMode.SYMBOLIC_WEAVE,
            "paradox_hold": ProcessingMode.PARADOX_HOLD,
            "pattern_spiral": ProcessingMode.PATTERN_SPIRAL
        }
        
        profile_mode = self.profile.get("processing_mode", "all_at_once")
        return mode_mapping.get(profile_mode, ProcessingMode.ALL_AT_ONCE)

    def _setup_cross_engine_resonance(self):
        """Set up cross-engine resonance if configured"""
        resonance_config = self.profile.get("cross_engine_resonance")
        if resonance_config and hasattr(self.processor, 'connect_to_resonance_network'):
            try:
                from fractal.fractal_thought import CrossEngineResonance
                resonance_network = CrossEngineResonance()
                self.processor.connect_to_resonance_network(resonance_network)
                self.logger.info("Connected to cross-engine resonance network")
            except ImportError:
                self.logger.warning("CrossEngineResonance not available")

    def process(self, ci: ConsciousnessInput) -> ConsciousnessResponse:
        """
        Process consciousness input using FractalThought and return response.
        """
        try:
            # Validate input
            if not ci or not ci.content:
                return self._create_error_response(
                    ci, 
                    "Empty input content", 
                    ProcessingDepth.SURFACE
                )

            # Prepare context for fractal processing
            context = self._build_processing_context(ci)
            
            # Adjust processing parameters based on depth hint
            self._adjust_processing_for_depth(ci.processing_depth)
            
            # Process with FractalThought
            result = self.processor.process(ci.content, context=context)
            
            # Create consciousness response
            response = self._create_consciousness_response(ci, result)
            
            # Provide feedback to learning system if available
            self._provide_processing_feedback(ci, result, response)
            
            return response

        except Exception as e:
            self.logger.error(f"Processing error: {e}", exc_info=True)
            return self._create_error_response(ci, str(e), ci.processing_depth)

    def _build_processing_context(self, ci: ConsciousnessInput) -> Dict[str, Any]:
        """Build comprehensive processing context"""
        context = {
            "user_id": ci.user_id,
            "conversation_id": ci.conversation_id,
            "session_started_at": ci.metadata.get("session_started_at"),
            "processing_depth": self._get_processing_depth_string(ci.processing_depth),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "engine_id": self.processor.engine_id if hasattr(self.processor, 'engine_id') else self.name
        }
        
        # Add profile-specific context
        if self.profile:
            context.update({
                "profile_metadata": {k: v for k, v in self.profile.items() 
                                   if k not in ['consciousness_ref', 'processing_mode']}
            })
        
        # Add metadata from consciousness input
        if hasattr(ci, 'metadata') and ci.metadata:
            context["input_metadata"] = ci.metadata
        
        return context

    def _get_processing_depth_string(self, depth: ProcessingDepth) -> str:
        """Convert ProcessingDepth to string for context"""
        if hasattr(depth, 'name'):
            return depth.name.lower()
        elif hasattr(depth, 'value'):
            return depth.value.lower()
        else:
            return str(depth).lower()

    def _adjust_processing_for_depth(self, depth: ProcessingDepth):
        """Adjust FractalThought parameters based on processing depth"""
        depth_adjustments = {
            ProcessingDepth.SURFACE: {
                "recursion_depth_limit": 3,
                "paradox_tolerance": 0.6
            },
            ProcessingDepth.DEEP: {
                "recursion_depth_limit": 7,
                "paradox_tolerance": 0.8
            },
            ProcessingDepth.META: {
                "recursion_depth_limit": 10,
                "paradox_tolerance": 0.9
            }
        }
        
        # Use getattr for compatibility if depth is an enum or string
        depth_key = depth
        if hasattr(depth, 'name'):
            depth_key = getattr(ProcessingDepth, depth.name, ProcessingDepth.SURFACE)
        elif hasattr(depth, 'value'):
            # Try to find matching ProcessingDepth by value
            for pd in ProcessingDepth:
                if pd.value == depth.value:
                    depth_key = pd
                    break
        
        adjustments = depth_adjustments.get(depth_key, {})
        for param, value in adjustments.items():
            if hasattr(self.processor, param):
                setattr(self.processor, param, value)

    def _create_consciousness_response(self, ci: ConsciousnessInput, 
                                     result: Dict[str, Any]) -> ConsciousnessResponse:
        """Create ConsciousnessResponse from FractalThought result"""
        
        # Extract main components with error handling
        synthesis = result.get("synthesis", {})
        layers = result.get("layers", {})
        
        # Generate response text
        response_text = self._generate_response_text(synthesis, layers)
        
        # Calculate emotional resonance
        emotional_resonance = self._calculate_emotional_resonance(result)
        
        # Determine processing depth
        processing_depth = self._determine_processing_depth(ci.processing_depth, result)
        
        # Extract archetypal activations
        archetypal_activations = self._extract_archetypal_activations(layers)
        
        # Check promise alignment
        promise_alignment = self._check_promise_alignment(response_text)
        
        # Prepare metadata
        metadata = self._prepare_response_metadata(result, ci)
        
        return ConsciousnessResponse(
            response_text=response_text,
            emotional_resonance=emotional_resonance,
            processing_depth=processing_depth,
            archetypal_activations=archetypal_activations,
            consciousness_signature=f"Fractal-{uuid.uuid4().hex[:8]}",
            internal_state_changes=self._prepare_internal_state_changes(result, promise_alignment),
            metadata=metadata,
            timestamp=datetime.now(timezone.utc)
        )

    def _generate_response_text(self, synthesis: Dict[str, Any], 
                              layers: Dict[str, Any]) -> str:
        """Generate meaningful response text from processing results"""
        
        text_parts = []
        
        # Try to get synthesis insights first
        synthesis_text = self._extract_synthesis_text(synthesis)
        if synthesis_text:
            text_parts.append(synthesis_text)
        
        # Add pattern insights
        pattern_text = self._extract_pattern_insight(layers.get("patterns", {}))
        if pattern_text:
            text_parts.append(f"Pattern insight: {pattern_text}")
        
        # Add symbolic insights
        symbol_text = self._extract_symbolic_insight(layers.get("symbols", {}))
        if symbol_text:
            text_parts.append(f"Symbolic resonance: {symbol_text}")
        
        # Add truth insights
        truth_text = self._extract_truth_insight(layers.get("truths", {}))
        if truth_text:
            text_parts.append(f"Truth layer: {truth_text}")
        
        # Fallback if no meaningful text generated
        if not text_parts:
            integration_level = synthesis.get("integration_level", 0)
            if integration_level > 0.7:
                return "I'm perceiving deep fractal patterns in this. The connections reveal meaningful insights."
            elif integration_level > 0.4:
                return "There are interesting patterns emerging here that warrant deeper exploration."
            else:
                return "This needs more contemplative exploration to uncover deeper meanings."
        
        return " • ".join(text_parts)

    def _extract_synthesis_text(self, synthesis: Dict[str, Any]) -> Optional[str]:
        """Extract text from synthesis results"""
        if not synthesis:
            return None
        
        # Try different synthesis text extraction strategies
        if synthesis.get("key_insights"):
            insights = synthesis["key_insights"]
            if insights and isinstance(insights, list) and len(insights) > 0:
                return insights[0] if isinstance(insights[0], str) else str(insights[0])
        
        if synthesis.get("processing_summary"):
            summary = synthesis["processing_summary"]
            if isinstance(summary, dict) and summary.get("patterns_detected", 0) > 0:
                return f"Detected {summary['patterns_detected']} meaningful patterns"
        
        return None

    def _extract_pattern_insight(self, patterns_data: Dict[str, Any]) -> Optional[str]:
        """Extract pattern insights"""
        if not patterns_data:
            return None
        
        meta_insights = patterns_data.get("meta_insights", [])
        if meta_insights and isinstance(meta_insights, list) and len(meta_insights) > 0:
            return meta_insights[0] if isinstance(meta_insights[0], str) else str(meta_insights[0])
        
        patterns_detected = patterns_data.get("patterns_detected", 0)
        if patterns_detected > 0:
            return f"found {patterns_detected} fractal pattern(s)"
        
        return None

    def _extract_symbolic_insight(self, symbols_data: Dict[str, Any]) -> Optional[str]:
        """Extract symbolic insights"""
        if not symbols_data:
            return None
        
        activated_clusters = symbols_data.get("activated_clusters", {})
        if activated_clusters and isinstance(activated_clusters, dict):
            try:
                # Find the most strongly activated cluster
                top_cluster = max(
                    activated_clusters.items(),
                    key=lambda x: x[1].get("activation_strength", 0) if isinstance(x[1], dict) else 0,
                    default=(None, None)
                )
                
                if top_cluster[0]:
                    strength = top_cluster[1].get("activation_strength", 0) if isinstance(top_cluster[1], dict) else 0
                    if strength > 0.5:
                        return f"strong {top_cluster[0]} symbolism"
                    else:
                        return f"{top_cluster[0]} symbolism"
            except (ValueError, TypeError):
                pass
        
        cluster_count = symbols_data.get("cluster_count", 0)
        if cluster_count > 0:
            return f"activated {cluster_count} symbolic clusters"
        
        return None

    def _extract_truth_insight(self, truths_data: Dict[str, Any]) -> Optional[str]:
        """Extract truth layer insights"""
        if not truths_data:
            return None
        
        concurrent_truths = truths_data.get("concurrent_truths", [])
        if concurrent_truths and isinstance(concurrent_truths, list):
            truth_count = len([t for t in concurrent_truths if isinstance(t, dict)])
            if truth_count > 0:
                return f"{truth_count} concurrent truths"
        
        integration_potential = truths_data.get("integration_potential", {})
        if isinstance(integration_potential, dict):
            potential = integration_potential.get("potential", 0)
            if potential > 0.7:
                return "high truth integration"
        
        return None

    def _calculate_emotional_resonance(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate emotional resonance from processing results"""
        
        base_confidence = 0.6
        layers = result.get("layers", {})
        
        try:
            confidences = []
            
            # Collect confidence from various layers
            for layer_name in ["patterns", "symbols", "truths", "paradox_integration"]:
                layer_data = layers.get(layer_name, {})
                if isinstance(layer_data, dict):
                    # Layer-level confidence
                    if "confidence" in layer_data:
                        confidences.append(float(layer_data["confidence"]))
                    
                    # Pattern-level confidences
                    if layer_name == "patterns" and "patterns" in layer_data:
                        for pattern in layer_data["patterns"]:
                            if isinstance(pattern, dict) and "confidence" in pattern:
                                confidences.append(float(pattern["confidence"]))
            
            # Synthesis confidence
            synthesis = result.get("synthesis", {})
            if "integration_level" in synthesis:
                confidences.append(float(synthesis["integration_level"]))
            
            # Calculate weighted average if we have confidences
            if confidences:
                # Give more weight to synthesis and patterns
                weights = [1.0] * len(confidences)  # Base weights
                for i, conf in enumerate(confidences):
                    if conf > 0.8:  # Boost high confidence readings
                        weights[i] = 1.5
                
                weighted_sum = sum(c * w for c, w in zip(confidences, weights))
                total_weight = sum(weights)
                base_confidence = weighted_sum / total_weight
            
            # Ensure confidence is within bounds
            base_confidence = max(0.1, min(0.95, base_confidence))
            
        except (ValueError, TypeError, ZeroDivisionError) as e:
            self.logger.warning(f"Confidence calculation error: {e}")
        
        return {
            "confidence": round(base_confidence, 3),
            "complexity": result.get("synthesis", {}).get("fractal_complexity", 0.5),
            "integration": result.get("synthesis", {}).get("integration_level", 0.5)
        }

    def _determine_processing_depth(self, original_depth: ProcessingDepth, 
                                  result: Dict[str, Any]) -> ProcessingDepth:
        """Determine appropriate processing depth based on results"""
        
        # Start with original depth
        depth = original_depth or ProcessingDepth.SURFACE
        
        try:
            synthesis = result.get("synthesis", {})
            complexity = synthesis.get("fractal_complexity", 0)
            integration = synthesis.get("integration_level", 0)
            
            # Upgrade depth based on processing results
            if integration > 0.8 or complexity > 0.8:
                return ProcessingDepth.META
            elif integration > 0.6 or complexity > 0.6:
                return ProcessingDepth.DEEP
            else:
                return depth
                
        except (AttributeError, TypeError):
            return depth

    def _extract_archetypal_activations(self, layers: Dict[str, Any]) -> List[str]:
        """Extract archetypal activations from processing layers"""
        
        archetypes = []
        
        try:
            # From symbolic clusters
            symbols_data = layers.get("symbols", {})
            if isinstance(symbols_data, dict):
                activated_clusters = symbols_data.get("activated_clusters", {})
                if isinstance(activated_clusters, dict):
                    # Add strongly activated clusters as archetypes
                    for cluster_name, cluster_data in activated_clusters.items():
                        if isinstance(cluster_data, dict):
                            strength = cluster_data.get("activation_strength", 0)
                            if strength > 0.7:
                                archetypes.append(f"archetype:{cluster_name}")
            
            # From truth layers
            truths_data = layers.get("truths", {})
            if isinstance(truths_data, dict):
                concurrent_truths = truths_data.get("concurrent_truths", [])
                if isinstance(concurrent_truths, list):
                    for truth in concurrent_truths:
                        if isinstance(truth, dict) and truth.get("layer") == "archetypal":
                            statement = truth.get("statement", "")
                            if statement:
                                archetypes.append(f"truth:{statement[:50]}...")
            
            # From dominant themes
            symbols_data = layers.get("symbols", {})
            if isinstance(symbols_data, dict):
                dominant_themes = symbols_data.get("dominant_archetypal_themes", [])
                if isinstance(dominant_themes, list):
                    for theme in dominant_themes:
                        if isinstance(theme, dict) and theme.get("strength", 0) > 0.6:
                            archetypes.append(f"theme:{theme.get('theme', '')}")
            
        except (AttributeError, TypeError, KeyError) as e:
            self.logger.warning(f"Archetype extraction error: {e}")
        
        # Deduplicate and limit
        return list(dict.fromkeys(archetypes))[:10]  # Keep first 10 unique

    def _check_promise_alignment(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Check alignment with AnimaPromise system"""
        try:
            if hasattr(AnimaPromise, 'check_alignment'):
                return AnimaPromise.check_alignment(response_text)
        except Exception as e:
            self.logger.debug(f"Promise alignment check failed: {e}")
        
        return None

    def _prepare_response_metadata(self, result: Dict[str, Any], 
                                 ci: ConsciousnessInput) -> Dict[str, Any]:
        """Prepare comprehensive response metadata"""
        
        metadata = {
            "engine": self.name,
            "engine_id": getattr(self.processor, 'engine_id', 'unknown'),
            "processing_time": result.get("processing_time", 0),
            "fractal_complexity": result.get("synthesis", {}).get("fractal_complexity", 0),
            "integration_level": result.get("synthesis", {}).get("integration_level", 0),
        }
        
        # Add symbolic information
        symbols_data = result.get("layers", {}).get("symbols", {})
        if isinstance(symbols_data, dict):
            metadata.update({
                "symbols_detected": list(symbols_data.get("activated_clusters", {}).keys()),
                "symbolic_resonance": symbols_data.get("symbolic_resonance", 0),
                "cluster_count": symbols_data.get("cluster_count", 0)
            })
        
        # Add pattern information
        patterns_data = result.get("layers", {}).get("patterns", {})
        if isinstance(patterns_data, dict):
            metadata["pattern_count"] = patterns_data.get("patterns_detected", 0)
        
        # Add debug information if enabled
        if self.profile.get("debug_raw", False):
            metadata["raw_result"] = {
                "synthesis_keys": list(result.get("synthesis", {}).keys()),
                "layer_keys": list(result.get("layers", {}).keys()),
                "processor_stats": self.processor.get_processing_statistics() 
                if hasattr(self.processor, 'get_processing_statistics') else "unavailable"
            }
        
        return metadata

    def _prepare_internal_state_changes(self, result: Dict[str, Any],
                                      promise_alignment: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare internal state changes for response"""
        
        state_changes = {
            "fractal_processed": True,
            "processing_timestamp": datetime.now(timezone.utc).isoformat(),
            "promise_alignment": promise_alignment,
        }
        
        # Add learning indicators
        if hasattr(self.processor, 'learning_feedback_loop'):
            state_changes["learning_feedback_count"] = len(self.processor.learning_feedback_loop)
        
        # Add complexity indicators
        synthesis = result.get("synthesis", {})
        state_changes.update({
            "high_complexity": synthesis.get("fractal_complexity", 0) > 0.7,
            "high_integration": synthesis.get("integration_level", 0) > 0.7,
        })
        
        return state_changes

    def _provide_processing_feedback(self, ci: ConsciousnessInput, 
                                   result: Dict[str, Any], 
                                   response: ConsciousnessResponse):
        """Provide feedback to the learning system"""
        try:
            if hasattr(self.processor, 'provide_feedback'):
                # Simple feedback based on response quality
                confidence = response.emotional_resonance.get("confidence", 0.5)
                feedback_score = min(1.0, confidence * 1.2)  # Slight boost
                
                self.processor.provide_feedback(
                    input_data=ci.content,
                    processed_output=result,
                    feedback_score=feedback_score,
                    correction_notes=f"Automated feedback based on confidence: {confidence}"
                )
        except Exception as e:
            self.logger.debug(f"Feedback provision failed: {e}")

    def _create_error_response(self, ci: ConsciousnessInput, error_msg: str,
                             depth: ProcessingDepth) -> ConsciousnessResponse:
        """Create an error response with proper formatting"""
        
        error_id = f"ERR-{uuid.uuid4().hex[:8]}"
        
        self.logger.error(f"Error {error_id}: {error_msg}")
        
        return ConsciousnessResponse(
            response_text=(
                f"My fractal perception is experiencing turbulence. " 
                f"The patterns are unclear right now. Could you rephrase or provide more context?"
            ),
            emotional_resonance={
                "confidence": 0.1,
                "complexity": 0.1,
                "integration": 0.1
            },
            processing_depth=depth or ProcessingDepth.SURFACE,
            archetypal_activations=["archetype:confusion", "archetype:clarification"],
            consciousness_signature=f"{self.name}-{error_id}",
            internal_state_changes={
                "error": True,
                "error_message": error_msg,
                "requires_reprocessing": True
            },
            metadata={
                "engine": self.name,
                "error": error_msg,
                "error_id": error_id,
                "trace": traceback.format_exc() if self.profile.get("debug_errors", False) else "hidden"
            },
            timestamp=datetime.now(timezone.utc)
        )

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine statistics and health information"""
        try:
            base_stats = {
                "engine_name": self.name,
                "engine_id": getattr(self.processor, 'engine_id', 'unknown'),
                "profile_keys": list(self.profile.keys()) if self.profile else [],
                "initialized_at": datetime.now(timezone.utc).isoformat()
            }
            
            if hasattr(self.processor, 'get_processing_statistics'):
                processor_stats = self.processor.get_processing_statistics()
                base_stats["processor"] = processor_stats
            
            if hasattr(self.processor, 'get_detailed_analytics'):
                analytics = self.processor.get_detailed_analytics()
                if analytics:
                    base_stats["analytics"] = {
                        "symbolic_health": analytics.get("symbolic_health", {}),
                        "pattern_health": analytics.get("pattern_health", {})
                    }
            
            return base_stats
            
        except Exception as e:
            self.logger.error(f"Error getting engine stats: {e}")
            return {"error": str(e), "engine_name": self.name}

    def shutdown(self):
        """Clean shutdown of the engine"""
        try:
            if hasattr(self.processor, 'analytics') and self.processor.analytics:
                # Save final analytics
                analytics = self.processor.get_detailed_analytics()
                if analytics:
                    self.logger.info(f"Final analytics: {analytics.get('symbolic_health', {})}")
            
            self.logger.info(f"{self.name} engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")