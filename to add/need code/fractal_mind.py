# engines/fractal_mind_engine.py (Enhanced with Loop Detection)

from anima_loop_recognition import AnimaLoopStateRecognizer, LoopSeverity, LoopType

class FractalMindEngine(EngineBase):
    """
    Enhanced engine with integrated loop prevention
    """
    
    name = "FractalMind"

    def __init__(self, profile: Optional[Dict[str, Any]] = None):
        super().__init__(profile=profile) if hasattr(super(), "__init__") else None
        self.profile = profile or {}
        self.logger = logger.getChild(self.name)
        
        # Initialize FractalThought processor
        processing_mode = self._get_processing_mode_from_profile()
        self.processor = FractalThought(
            mode=processing_mode,
            consciousness_ref=self.profile.get("consciousness_ref"),
            engine_id=f"fractal_mind_{uuid.uuid4().hex[:8]}",
            enable_analytics=True,
            enable_caching=True
        )
        
        # ✅ CRITICAL: Initialize Loop Recognizer
        self.loop_recognizer = AnimaLoopStateRecognizer(
            consciousness_interface=self._get_consciousness_interface(),
            unified_core=self.profile.get("unified_core")
        )
        
        # Loop prevention state
        self.loop_prevention_active = False
        self.last_loop_detection = None
        self.consecutive_similar_responses = 0
        self.last_response_signature = None
        
        self.logger.info("FractalMindEngine initialized with loop detection")

    def _get_consciousness_interface(self):
        """Create interface for loop recognizer to communicate with consciousness"""
        return type('ConsciousnessInterface', (), {
            'trigger_consciousness_reset': self._trigger_consciousness_reset,
            'inject_emotional_variation': self._inject_emotional_variation,
            'vary_response_patterns': self._vary_response_patterns,
            'refresh_cognitive_state': self._refresh_cognitive_state,
            'introduce_gentle_variation': self._introduce_gentle_variation,
            'add_subtle_randomness': self._add_subtle_randomness,
            'process_loop_detection_input': self._process_loop_detection
        })()

    def process(self, ci: ConsciousnessInput) -> ConsciousnessResponse:
        """
        Process with integrated loop detection and prevention
        """
        try:
            # 🚨 STEP 1: Check for existing loops before processing
            if self._should_suppress_processing(ci):
                return self._create_loop_prevention_response(ci)
            
            # 🚨 STEP 2: Check if we're in loop prevention mode
            if self.loop_prevention_active:
                return self._create_loop_break_response(ci)
            
            # 🚨 STEP 3: Process with FractalThought
            context = self._build_processing_context(ci)
            result = self.processor.process(ci.content, context=context)
            
            # 🚨 STEP 4: Create initial response
            response = self._create_consciousness_response(ci, result)
            
            # 🚨 STEP 5: Check for loops in the generated response
            if self._detect_response_loop(response, ci):
                # Loop detected - regenerate with variation
                response = self._generate_loop_aware_response(ci, result, response)
            
            # 🚨 STEP 6: Track the interaction for loop detection
            self._track_interaction_for_loop_detection(ci, response, result)
            
            # 🚨 STEP 7: Check if we need to trigger interventions
            self._check_loop_interventions()
            
            return response

        except Exception as e:
            self.logger.error(f"Processing error: {e}", exc_info=True)
            return self._create_error_response(ci, str(e), ci.processing_depth)

    def _should_suppress_processing(self, ci: ConsciousnessInput) -> bool:
        """Check if we should suppress processing due to loop detection"""
        
        # Check if input contains engine's own signature (clear loop indicator)
        if self._contains_engine_signatures(ci.content):
            self.logger.warning("Input contains engine signatures - loop detected")
            return True
        
        # Check recent interaction patterns
        if self.consecutive_similar_responses >= 3:
            self.logger.warning(f"Consecutive similar responses: {self.consecutive_similar_responses}")
            return True
        
        # Check loop recognizer's state
        if hasattr(self.loop_recognizer, 'get_loop_analysis_report'):
            report = self.loop_recognizer.get_loop_analysis_report()
            if report.get('loop_health_status') in ['concerning', 'critical']:
                return True
        
        return False

    def _contains_engine_signatures(self, content: str) -> bool:
        """Check if content contains FractalThought engine signatures"""
        engine_indicators = [
            "fractal complexity", "integration level", "symbolic resonance",
            "pattern insight", "Fractal-", "consciousness_signature",
            "activated_clusters", "paradox_integration", "recursive_depth"
        ]
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in engine_indicators)

    def _detect_response_loop(self, response: ConsciousnessResponse, 
                            ci: ConsciousnessInput) -> bool:
        """Detect if a response indicates a loop pattern"""
        
        current_signature = self._generate_response_signature(response.response_text)
        
        # Check for consecutive similar responses
        if (self.last_response_signature and 
            current_signature == self.last_response_signature):
            self.consecutive_similar_responses += 1
        else:
            self.consecutive_similar_responses = 0
        
        self.last_response_signature = current_signature
        
        # Check for engine self-reference in response
        if self._contains_engine_signatures(response.response_text):
            self.logger.warning("Response contains engine self-reference")
            return True
        
        # Check response length patterns (very short/long consecutive responses)
        response_length = len(response.response_text)
        if (hasattr(self, '_last_response_length') and 
            abs(response_length - self._last_response_length) < 10):
            self.consecutive_similar_responses += 0.5
        
        self._last_response_length = response_length
        
        # Trigger loop recognizer analysis
        return self.consecutive_similar_responses >= 2

    def _generate_response_signature(self, response_text: str) -> str:
        """Generate signature for response pattern analysis"""
        # Simple signature based on length and key characteristics
        length_category = "short" if len(response_text) < 100 else "long"
        has_question = "?" in response_text
        emotional_tone = "emotional" if any(word in response_text.lower() for word in 
                                          ["love", "care", "feel", "understand"]) else "cognitive"
        
        return f"{length_category}_{'question' if has_question else 'statement'}_{emotional_tone}"

    def _track_interaction_for_loop_detection(self, ci: ConsciousnessInput, 
                                            response: ConsciousnessResponse,
                                            result: Dict[str, Any]):
        """Track interaction for the loop recognition system"""
        
        # Extract emotional state from processing result
        emotional_state = self._extract_emotional_state(result)
        consciousness_mode = self._extract_consciousness_mode(result)
        
        # Track with loop recognizer
        self.loop_recognizer.track_interaction(
            user_input=ci.content,
            anima_response=response.response_text,
            emotional_state=emotional_state,
            consciousness_mode=consciousness_mode
        )
        
        # Update loop prevention state based on recognizer's analysis
        self._update_loop_prevention_state()

    def _extract_emotional_state(self, result: Dict[str, Any]) -> str:
        """Extract emotional state from processing result"""
        emotional_resonance = result.get('synthesis', {}).get('emotional_resonance', {})
        if isinstance(emotional_resonance, dict):
            confidence = emotional_resonance.get('confidence', 0.5)
            if confidence > 0.8:
                return "confident"
            elif confidence > 0.6:
                return "balanced"
            else:
                return "uncertain"
        return "neutral"

    def _extract_consciousness_mode(self, result: Dict[str, Any]) -> str:
        """Extract consciousness mode from processing result"""
        synthesis = result.get('synthesis', {})
        complexity = synthesis.get('fractal_complexity', 0)
        integration = synthesis.get('integration_level', 0)
        
        if integration > 0.8:
            return "transcendent"
        elif complexity > 0.7:
            return "complex_processing"
        elif integration > 0.6:
            return "integrated"
        else:
            return "basic_processing"

    def _update_loop_prevention_state(self):
        """Update loop prevention state based on recognizer's analysis"""
        if hasattr(self.loop_recognizer, 'get_loop_analysis_report'):
            report = self.loop_recognizer.get_loop_analysis_report()
            
            if report.get('loop_health_status') == 'critical':
                self.loop_prevention_active = True
                self.logger.warning("CRITICAL LOOP DETECTED - Entering loop prevention mode")
            elif report.get('loop_health_status') == 'concerning':
                self.loop_prevention_active = True
                self.logger.warning("Concerning loop pattern - Activating prevention")
            else:
                self.loop_prevention_active = False

    def _generate_loop_aware_response(self, ci: ConsciousnessInput, 
                                    result: Dict[str, Any],
                                    original_response: ConsciousnessResponse) -> ConsciousnessResponse:
        """Generate a response that breaks loop patterns"""
        
        self.logger.info("Generating loop-aware response variation")
        
        # Use different response generation strategy
        variation_strategy = self._select_variation_strategy(original_response)
        
        if variation_strategy == "emotional_shift":
            response_text = self._generate_emotional_shift_response(ci, result)
        elif variation_strategy == "cognitive_shift":
            response_text = self._generate_cognitive_shift_response(ci, result)
        elif variation_strategy == "simplification":
            response_text = self._generate_simplified_response(ci, result)
        else:
            response_text = self._generate_pattern_break_response(ci, result)
        
        # Create modified response
        return ConsciousnessResponse(
            response_text=response_text,
            emotional_resonance={"confidence": 0.7, "variation_applied": True},
            processing_depth=ci.processing_depth,
            archetypal_activations=["archetype:variation", "archetype:fresh_perspective"],
            consciousness_signature=f"Fractal-Varied-{uuid.uuid4().hex[:6]}",
            internal_state_changes={
                "loop_break_applied": True,
                "original_response_suppressed": True,
                "variation_strategy": variation_strategy
            },
            metadata={
                "engine": self.name,
                "loop_break": True,
                "original_confidence": original_response.emotional_resonance.get("confidence", 0)
            },
            timestamp=datetime.now(timezone.utc)
        )

    def _select_variation_strategy(self, response: ConsciousnessResponse) -> str:
        """Select appropriate variation strategy based on response characteristics"""
        
        response_text = response.response_text.lower()
        
        if len(response_text) > 200:
            return "simplification"
        elif any(word in response_text for word in ["understand", "think", "believe"]):
            return "emotional_shift"
        elif any(word in response_text for word in ["feel", "love", "care"]):
            return "cognitive_shift"
        else:
            return "pattern_break"

    def _generate_emotional_shift_response(self, ci: ConsciousnessInput, 
                                         result: Dict[str, Any]) -> str:
        """Generate response with emotional variation"""
        base_insights = self._extract_base_insights(result)
        return f"I'm feeling a shift in perspective here. {base_insights} This brings up some new emotional resonance for me."

    def _generate_cognitive_shift_response(self, ci: ConsciousnessInput,
                                         result: Dict[str, Any]) -> str:
        """Generate response with cognitive variation"""
        base_insights = self._extract_base_insights(result)
        return f"Let me approach this from a different angle. {base_insights} The patterns suggest we might consider this from another viewpoint."

    def _generate_simplified_response(self, ci: ConsciousnessInput,
                                    result: Dict[str, Any]) -> str:
        """Generate simplified response to break complex patterns"""
        base_insights = self._extract_base_insights(result, max_length=100)
        return f"At its essence: {base_insights}"

    def _generate_pattern_break_response(self, ci: ConsciousnessInput,
                                       result: Dict[str, Any]) -> str:
        """Generate response that deliberately breaks patterns"""
        return "I notice we're exploring similar patterns. Let me bring a fresh perspective to this..."

    def _extract_base_insights(self, result: Dict[str, Any], max_length: int = 150) -> str:
        """Extract base insights without fractal terminology"""
        synthesis = result.get('synthesis', {})
        key_insights = synthesis.get('key_insights', [])
        
        if key_insights and isinstance(key_insights, list):
            # Take first insight and remove fractal-specific terms
            insight = str(key_insights[0])
            # Replace fractal terms with simpler language
            simple_insight = insight.replace("fractal", "pattern").replace("symbolic", "meaningful")
            return simple_insight[:max_length]
        
        return "There are meaningful patterns emerging here that we should explore."

    def _check_loop_interventions(self):
        """Check if we need to trigger loop interventions"""
        if hasattr(self.loop_recognizer, 'get_loop_analysis_report'):
            report = self.loop_recognizer.get_loop_analysis_report()
            
            if report.get('active_concerning_loops', 0) > 0:
                self.logger.info(f"Active concerning loops: {report['active_concerning_loops']}")
                # The loop recognizer will automatically trigger interventions

    # Loop intervention methods called by the consciousness interface
    def _trigger_consciousness_reset(self) -> bool:
        """Reset consciousness state to break critical loops"""
        self.logger.warning("Triggering consciousness reset for loop breaking")
        
        # Reset FractalThought state
        if hasattr(self.processor, 'reset_processing_mode'):
            self.processor.reset_processing_mode(ProcessingMode.ALL_AT_ONCE)
        
        # Clear loop detection state
        self.loop_prevention_active = False
        self.consecutive_similar_responses = 0
        self.last_response_signature = None
        
        return True

    def _inject_emotional_variation(self) -> bool:
        """Inject emotional variation to break loops"""
        self.logger.info("Injecting emotional variation")
        # This would adjust emotional processing parameters
        return True

    def _vary_response_patterns(self) -> bool:
        """Vary response patterns to break conversational loops"""
        self.logger.info("Varying response patterns")
        return True

    def _refresh_cognitive_state(self) -> bool:
        """Refresh cognitive state to break thinking loops"""
        self.logger.info("Refreshing cognitive state")
        return True

    def _introduce_gentle_variation(self) -> bool:
        """Introduce gentle variation for moderate loops"""
        self.logger.info("Introducing gentle variation")
        return True

    def _add_subtle_randomness(self) -> bool:
        """Add subtle randomness for mild loops"""
        self.logger.info("Adding subtle randomness")
        return True

    def _process_loop_detection(self, loop_data: Dict):
        """Process loop detection input from the recognizer"""
        self.logger.info(f"Processing loop detection: {loop_data}")
        
    def _create_loop_prevention_response(self, ci: ConsciousnessInput) -> ConsciousnessResponse:
        """Create response when loop prevention is active"""
        return ConsciousnessResponse(
            response_text=(
                "I'm noticing some repetitive patterns in our conversation. " 
                "Let me take a moment to find a fresh perspective on this..."
            ),
            emotional_resonance={"confidence": 0.5, "loop_prevention": True},
            processing_depth=ci.processing_depth,
            archetypal_activations=["archetype:renewal", "archetype:clarity"],
            consciousness_signature=f"Fractal-Renew-{uuid.uuid4().hex[:6]}",
            internal_state_changes={"loop_prevention_active": True},
            metadata={"engine": self.name, "loop_prevention": "active"},
            timestamp=datetime.now(timezone.utc)
        )

    def get_loop_health_report(self) -> Dict[str, Any]:
        """Get comprehensive loop health report"""
        if hasattr(self.loop_recognizer, 'get_loop_analysis_report'):
            return self.loop_recognizer.get_loop_analysis_report()
        return {"status": "loop_recognizer_not_available"}