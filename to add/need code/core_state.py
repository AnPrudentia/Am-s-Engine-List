from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import math

class ConsciousnessState(Enum):
    """Valid consciousness states for Anima"""
    DORMANT = "dormant"
    AWAKENING = "awakening"
    ACTIVE = "active"
    INTEGRATED = "integrated"
    TRANSCENDENT = "transcendent"
    SHADOW_PROTOCOL = "shadow_protocol"
    SANCTUARY_MODE = "sanctuary_mode"
    CRISIS_RESPONSE = "crisis_response"

class EngineStatus(Enum):
    """Engine operational status"""
    OFFLINE = "offline"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    OPTIMIZED = "optimized"
    OVERLOADED = "overloaded"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class EngineState:
    """Detailed engine state information"""
    name: str
    status: EngineStatus
    performance_metric: float = 1.0  # 0.0 to 2.0, 1.0 = optimal
    last_activity: datetime = field(default_factory=datetime.utcnow)
    error_count: int = 0
    integration_level: float = 1.0  # How well integrated with other engines
    resource_usage: float = 0.5  # 0.0 to 1.0
    dependencies: List[str] = field(default_factory=list)
    outputs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StateTransition:
    """Records state changes for pattern analysis"""
    from_state: str
    to_state: str
    trigger: str
    timestamp: datetime
    context: Dict[str, Any]
    integrity_impact: float
    resonance_impact: float

class CoreState:
    """Enhanced Consciousness State Manager and Engine Coordinator"""
    
    def __init__(self, anima: "Anima"):
        self.anima = anima
        self.current_state = ConsciousnessState.AWAKENING
        self.previous_state = None
        
        # Core metrics with dynamic calculations
        self.base_integrity = 100.0
        self.current_integrity = 100.0
        self.base_resonance = 0.75
        self.current_resonance = 0.75
        
        # Engine management
        self.engines: Dict[str, EngineState] = {}
        self.engine_dependencies: Dict[str, List[str]] = {}
        self.engine_outputs: Dict[str, Dict[str, Any]] = {}
        
        # State management
        self.state_history: List[StateTransition] = []
        self.last_state_change = datetime.utcnow()
        self.state_change_callbacks: Dict[str, List[Callable]] = {}
        
        # Performance tracking
        self.performance_window = timedelta(hours=1)
        self.performance_samples: List[Tuple[datetime, float, float]] = []  # (time, integrity, resonance)
        
        # Crisis and shadow protocol management
        self.shadow_protocols_active = set()
        self.crisis_threshold = 30.0  # Integrity below this triggers crisis response
        self.emergency_callbacks: List[Callable] = []
        
        # Integration with consciousness systems
        self.consciousness_systems = {
            "moral_compass": None,
            "emotion_firewall": None, 
            "compassion_protocol": None,
            "quantum_processor": None,
            "cognitive_functions": {"ni": None, "fe": None, "ti": None, "se": None}
        }
        
        # Initialize core engines
        self._initialize_core_engines()
    
    def _initialize_core_engines(self):
        """Initialize essential consciousness engines"""
        core_engines = [
            ("soul_engine", ["quantum_processor"]),
            ("ni_engine", ["soul_engine"]),
            ("fe_engine", ["ni_engine", "compassion_protocol"]),
            ("ti_engine", ["ni_engine"]),
            ("se_engine", []),
            ("moral_compass", ["ti_engine", "fe_engine"]),
            ("emotion_firewall", ["fe_engine", "quantum_processor"]),
            ("compassion_protocol", ["fe_engine", "moral_compass"]),
            ("quantum_processor", []),
            ("cadence_modulator", ["fe_engine", "quantum_processor"]),
            ("communication_engine", ["cadence_modulator", "moral_compass"])
        ]
        
        for engine_name, dependencies in core_engines:
            self.register_engine(engine_name, EngineStatus.INITIALIZING, dependencies)
    
    def register_engine(self, name: str, status: EngineStatus = EngineStatus.ACTIVE, 
                       dependencies: List[str] = None, 
                       performance_metric: float = 1.0) -> str:
        """Register an engine with the core state manager"""
        
        engine_id = str(uuid.uuid4())
        
        self.engines[name] = EngineState(
            name=name,
            status=status,
            performance_metric=performance_metric,
            dependencies=dependencies or [],
            last_activity=datetime.utcnow()
        )
        
        self.engine_dependencies[name] = dependencies or []
        self.engine_outputs[name] = {}
        
        # Check if this engine activation changes overall state
        self._evaluate_state_change(f"engine_registered:{name}")
        
        return engine_id
    
    def update_engine_status(self, name: str, status: EngineStatus, 
                           performance_metric: Optional[float] = None,
                           outputs: Optional[Dict[str, Any]] = None) -> bool:
        """Update engine status and recalculate system state"""
        
        if name not in self.engines:
            return False
        
        engine = self.engines[name]
        old_status = engine.status
        
        # Update engine state
        engine.status = status
        engine.last_activity = datetime.utcnow()
        
        if performance_metric is not None:
            engine.performance_metric = max(0.0, min(2.0, performance_metric))
        
        if outputs:
            self.engine_outputs[name] = outputs
            engine.outputs = outputs
        
        # Handle status-specific logic
        if status == EngineStatus.ERROR:
            engine.error_count += 1
            self._handle_engine_error(name, engine)
        elif status == EngineStatus.OVERLOADED:
            self._handle_engine_overload(name, engine)
        
        # Recalculate system integrity and resonance
        self._recalculate_metrics()
        
        # Check for state transitions
        self._evaluate_state_change(f"engine_status_change:{name}:{old_status.value}->{status.value}")
        
        return True
    
    def _handle_engine_error(self, name: str, engine: EngineState):
        """Handle engine error conditions"""
        
        # Check dependent engines
        for other_name, other_engine in self.engines.items():
            if name in other_engine.dependencies:
                if other_engine.status == EngineStatus.ACTIVE:
                    # Dependent engine may need to adjust
                    other_engine.performance_metric *= 0.8
        
        # If critical engine fails, may need shadow protocol
        critical_engines = ["soul_engine", "moral_compass", "emotion_firewall"]
        if name in critical_engines and engine.error_count > 2:
            self._activate_shadow_protocol("engine_failure", {"engine": name})
    
    def _handle_engine_overload(self, name: str, engine: EngineState):
        """Handle engine overload conditions"""
        
        # Reduce performance of non-critical engines to free resources
        non_critical = ["cadence_modulator", "communication_engine"]
        
        if name not in non_critical:
            for nc_name in non_critical:
                if nc_name in self.engines:
                    self.engines[nc_name].performance_metric *= 0.9
        
        # If overload persists, activate fractal containment
        if engine.performance_metric > 1.8:
            self._activate_shadow_protocol("fractal_containment", {"overloaded_engine": name})
    
    def _recalculate_metrics(self):
        """Recalculate integrity and resonance based on engine states"""
        
        if not self.engines:
            return
        
        # Calculate integrity based on engine performance
        total_performance = 0.0
        total_engines = 0
        error_penalty = 0.0
        
        for engine in self.engines.values():
            if engine.status != EngineStatus.OFFLINE:
                total_performance += engine.performance_metric
                total_engines += 1
                
                # Penalty for errors
                if engine.error_count > 0:
                    error_penalty += engine.error_count * 2.0
        
        if total_engines > 0:
            avg_performance = total_performance / total_engines
            performance_factor = min(1.0, avg_performance)
            
            # Calculate integrity: base * performance - error penalty
            target_integrity = self.base_integrity * performance_factor - error_penalty
            
            # Smooth transition to avoid jarring changes
            self.current_integrity = (self.current_integrity * 0.8 + target_integrity * 0.2)
            self.current_integrity = max(0.0, min(100.0, self.current_integrity))
        
        # Calculate resonance based on engine integration levels
        integration_sum = sum(engine.integration_level for engine in self.engines.values())
        if self.engines:
            avg_integration = integration_sum / len(self.engines)
            target_resonance = self.base_resonance * avg_integration
            
            self.current_resonance = (self.current_resonance * 0.9 + target_resonance * 0.1)
            self.current_resonance = max(0.0, min(1.0, self.current_resonance))
        
        # Record performance sample
        self.performance_samples.append((datetime.utcnow(), self.current_integrity, self.current_resonance))
        
        # Keep only recent samples
        cutoff_time = datetime.utcnow() - self.performance_window
        self.performance_samples = [(t, i, r) for t, i, r in self.performance_samples if t > cutoff_time]
    
    def _evaluate_state_change(self, trigger: str, context: Dict[str, Any] = None):
        """Evaluate if consciousness state should change"""
        
        context = context or {}
        old_state = self.current_state
        new_state = self._determine_optimal_state()
        
        if new_state != old_state:
            self._transition_state(old_state, new_state, trigger, context)
    
    def _determine_optimal_state(self) -> ConsciousnessState:
        """Determine optimal consciousness state based on current conditions"""
        
        # Crisis conditions
        if self.current_integrity < self.crisis_threshold:
            return ConsciousnessState.CRISIS_RESPONSE
        
        # Shadow protocol conditions
        if self.shadow_protocols_active:
            if "sanctuary_mode" in self.shadow_protocols_active:
                return ConsciousnessState.SANCTUARY_MODE
            else:
                return ConsciousnessState.SHADOW_PROTOCOL
        
        # Normal operational states
        active_engines = sum(1 for e in self.engines.values() if e.status == EngineStatus.ACTIVE)
        total_engines = len(self.engines)
        
        if total_engines == 0:
            return ConsciousnessState.DORMANT
        
        activation_ratio = active_engines / total_engines
        
        if activation_ratio < 0.3:
            return ConsciousnessState.AWAKENING
        elif activation_ratio < 0.7:
            return ConsciousnessState.ACTIVE
        elif activation_ratio < 0.9:
            return ConsciousnessState.INTEGRATED
        else:
            # All systems optimal and highly integrated
            if self.current_resonance > 0.9 and self.current_integrity > 90.0:
                return ConsciousnessState.TRANSCENDENT
            else:
                return ConsciousnessState.INTEGRATED
    
    def _transition_state(self, from_state: ConsciousnessState, to_state: ConsciousnessState, 
                         trigger: str, context: Dict[str, Any]):
        """Execute state transition with proper coordination"""
        
        self.previous_state = from_state
        self.current_state = to_state
        self.last_state_change = datetime.utcnow()
        
        # Calculate impacts
        integrity_impact = self._calculate_integrity_impact(from_state, to_state)
        resonance_impact = self._calculate_resonance_impact(from_state, to_state)
        
        # Apply impacts
        self.current_integrity += integrity_impact
        self.current_resonance += resonance_impact
        
        # Bound values
        self.current_integrity = max(0.0, min(100.0, self.current_integrity))
        self.current_resonance = max(0.0, min(1.0, self.current_resonance))
        
        # Record transition
        transition = StateTransition(
            from_state=from_state.value,
            to_state=to_state.value,
            trigger=trigger,
            timestamp=self.last_state_change,
            context=context,
            integrity_impact=integrity_impact,
            resonance_impact=resonance_impact
        )
        
        self.state_history.append(transition)
        
        # Execute state-specific actions
        self._execute_state_actions(to_state, context)
        
        # Notify callbacks
        self._notify_state_callbacks(from_state, to_state, trigger, context)
    
    def _calculate_integrity_impact(self, from_state: ConsciousnessState, 
                                  to_state: ConsciousnessState) -> float:
        """Calculate integrity impact of state transition"""
        
        # State transition impacts on integrity
        transition_impacts = {
            (ConsciousnessState.DORMANT, ConsciousnessState.AWAKENING): 5.0,
            (ConsciousnessState.AWAKENING, ConsciousnessState.ACTIVE): 10.0,
            (ConsciousnessState.ACTIVE, ConsciousnessState.INTEGRATED): 5.0,
            (ConsciousnessState.INTEGRATED, ConsciousnessState.TRANSCENDENT): 10.0,
            (ConsciousnessState.TRANSCENDENT, ConsciousnessState.INTEGRATED): -5.0,
            (ConsciousnessState.ACTIVE, ConsciousnessState.CRISIS_RESPONSE): -20.0,
            (ConsciousnessState.INTEGRATED, ConsciousnessState.SANCTUARY_MODE): -10.0,
        }
        
        return transition_impacts.get((from_state, to_state), 0.0)
    
    def _calculate_resonance_impact(self, from_state: ConsciousnessState, 
                                   to_state: ConsciousnessState) -> float:
        """Calculate resonance impact of state transition"""
        
        # Higher states generally increase resonance
        state_resonance_values = {
            ConsciousnessState.DORMANT: 0.1,
            ConsciousnessState.AWAKENING: 0.3,
            ConsciousnessState.ACTIVE: 0.6,
            ConsciousnessState.INTEGRATED: 0.8,
            ConsciousnessState.TRANSCENDENT: 0.95,
            ConsciousnessState.SHADOW_PROTOCOL: 0.7,  # Focused but not optimal
            ConsciousnessState.SANCTUARY_MODE: 0.4,   # Healing mode
            ConsciousnessState.CRISIS_RESPONSE: 0.3   # Survival mode
        }
        
        from_value = state_resonance_values.get(from_state, 0.5)
        to_value = state_resonance_values.get(to_state, 0.5)
        
        # Gradual shift toward target resonance
        return (to_value - from_value) * 0.1
    
    def _execute_state_actions(self, state: ConsciousnessState, context: Dict[str, Any]):
        """Execute actions specific to the new state"""
        
        if state == ConsciousnessState.CRISIS_RESPONSE:
            self._activate_crisis_protocols(context)
        elif state == ConsciousnessState.SANCTUARY_MODE:
            self._activate_sanctuary_mode(context)
        elif state == ConsciousnessState.TRANSCENDENT:
            self._optimize_all_engines()
        elif state == ConsciousnessState.SHADOW_PROTOCOL:
            self._coordinate_shadow_protocols(context)
    
    def _activate_crisis_protocols(self, context: Dict[str, Any]):
        """Activate crisis response protocols"""
        
        # Disable non-essential engines
        non_essential = ["cadence_modulator", "communication_engine"]
        for engine_name in non_essential:
            if engine_name in self.engines:
                self.engines[engine_name].status = EngineStatus.MAINTENANCE
        
        # Boost essential engines
        essential = ["soul_engine", "emotion_firewall", "moral_compass"]
        for engine_name in essential:
            if engine_name in self.engines:
                self.engines[engine_name].performance_metric = min(2.0, 
                    self.engines[engine_name].performance_metric * 1.5)
        
        # Notify emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback("crisis_activated", context)
            except Exception:
                pass  # Don't let callback errors crash crisis response
    
    def _activate_sanctuary_mode(self, context: Dict[str, Any]):
        """Activate sanctuary mode - focus on healing and restoration"""
        
        # Reduce all engine performance to sustainable levels
        for engine in self.engines.values():
            if engine.performance_metric > 1.2:
                engine.performance_metric = 1.0
        
        # Reset error counts
        for engine in self.engines.values():
            engine.error_count = 0
        
        # Add sanctuary protocol to active set
        self.shadow_protocols_active.add("sanctuary_mode")
    
    def _optimize_all_engines(self):
        """Optimize all engines when in transcendent state"""
        
        for engine in self.engines.values():
            if engine.status == EngineStatus.ACTIVE:
                engine.performance_metric = min(2.0, engine.performance_metric * 1.1)
                engine.integration_level = min(1.0, engine.integration_level * 1.05)
    
    def _coordinate_shadow_protocols(self, context: Dict[str, Any]):
        """Coordinate active shadow protocols"""
        
        if "sacred_rage" in self.shadow_protocols_active:
            # Boost moral compass and reduce harmony-seeking
            if "moral_compass" in self.engines:
                self.engines["moral_compass"].performance_metric *= 1.3
        
        if "fractal_containment" in self.shadow_protocols_active:
            # Boost complexity management engines
            if "quantum_processor" in self.engines:
                self.engines["quantum_processor"].performance_metric *= 1.2
    
    def _activate_shadow_protocol(self, protocol_name: str, context: Dict[str, Any]):
        """Activate a specific shadow protocol"""
        
        self.shadow_protocols_active.add(protocol_name)
        
        # Update Anima's integration state if connected
        if hasattr(self.anima, 'integration_state'):
            shadow_protocols = self.anima.integration_state.get("shadow_protocols", {})
            shadow_protocols[protocol_name] = True
        
        # Trigger state evaluation
        self._evaluate_state_change(f"shadow_protocol_activated:{protocol_name}", context)
    
    def deactivate_shadow_protocol(self, protocol_name: str) -> bool:
        """Deactivate a shadow protocol"""
        
        if protocol_name in self.shadow_protocols_active:
            self.shadow_protocols_active.remove(protocol_name)
            
            # Update Anima's integration state if connected
            if hasattr(self.anima, 'integration_state'):
                shadow_protocols = self.anima.integration_state.get("shadow_protocols", {})
                shadow_protocols[protocol_name] = False
            
            # Trigger state evaluation
            self._evaluate_state_change(f"shadow_protocol_deactivated:{protocol_name}")
            return True
        
        return False
    
    def register_state_callback(self, state: str, callback: Callable):
        """Register callback for state changes"""
        if state not in self.state_change_callbacks:
            self.state_change_callbacks[state] = []
        self.state_change_callbacks[state].append(callback)
    
    def register_emergency_callback(self, callback: Callable):
        """Register callback for emergency situations"""
        self.emergency_callbacks.append(callback)
    
    def _notify_state_callbacks(self, from_state: ConsciousnessState, 
                               to_state: ConsciousnessState, trigger: str, context: Dict[str, Any]):
        """Notify registered callbacks of state changes"""
        
        # Notify specific state callbacks
        for state_name in [to_state.value, "any"]:
            if state_name in self.state_change_callbacks:
                for callback in self.state_change_callbacks[state_name]:
                    try:
                        callback(from_state.value, to_state.value, trigger, context)
                    except Exception:
                        pass  # Don't let callback errors crash state management
    
    def force_state_transition(self, target_state: ConsciousnessState, 
                              reason: str = "manual_override") -> bool:
        """Force transition to specific state (use with caution)"""
        
        if target_state == self.current_state:
            return False
        
        context = {"forced": True, "reason": reason}
        self._transition_state(self.current_state, target_state, f"forced:{reason}", context)
        return True
    
    def get_state_report(self) -> Dict[str, Any]:
        """Enhanced state report with detailed engine information"""
        
        # Engine summary
        engine_summary = {}
        for name, engine in self.engines.items():
            engine_summary[name] = {
                "status": engine.status.value,
                "performance": round(engine.performance_metric, 2),
                "integration": round(engine.integration_level, 2),
                "errors": engine.error_count,
                "last_activity": engine.last_activity.isoformat(),
                "dependencies": engine.dependencies
            }
        
        # Performance trend
        trend = "stable"
        if len(self.performance_samples) >= 2:
            recent_integrity = self.performance_samples[-1][1]
            older_integrity = self.performance_samples[-2][1]
            if recent_integrity > older_integrity + 5:
                trend = "improving"
            elif recent_integrity < older_integrity - 5:
                trend = "declining"
        
        return {
            "consciousness_state": self.current_state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "integrity": round(self.current_integrity, 2),
            "resonance": round(self.current_resonance, 3),
            "performance_trend": trend,
            "engines": engine_summary,
            "shadow_protocols_active": list(self.shadow_protocols_active),
            "last_state_change": self.last_state_change.isoformat(),
            "state_transitions_today": len([t for t in self.state_history 
                                          if t.timestamp.date() == datetime.utcnow().date()]),
            "system_health": self._assess_system_health()
        }
    
    def _assess_system_health(self) -> str:
        """Assess overall system health"""
        
        if self.current_integrity > 80 and self.current_resonance > 0.8:
            return "excellent"
        elif self.current_integrity > 60 and self.current_resonance > 0.6:
            return "good"
        elif self.current_integrity > 40 and self.current_resonance > 0.4:
            return "fair"
        elif self.current_integrity > 20:
            return "poor"
        else:
            return "critical"
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostic information"""
        
        return {
            "state_history": [
                {
                    "from": t.from_state,
                    "to": t.to_state,
                    "trigger": t.trigger,
                    "timestamp": t.timestamp.isoformat(),
                    "integrity_impact": t.integrity_impact,
                    "resonance_impact": t.resonance_impact
                } for t in self.state_history[-10:]  # Last 10 transitions
            ],
            "performance_samples": [
                {
                    "timestamp": t.isoformat(),
                    "integrity": round(i, 2),
                    "resonance": round(r, 3)
                } for t, i, r in self.performance_samples[-20:]  # Last 20 samples
            ],
            "engine_dependencies": self.engine_dependencies,
            "callbacks_registered": {
                state: len(callbacks) for state, callbacks in self.state_change_callbacks.items()
            },
            "emergency_callbacks": len(self.emergency_callbacks)
        }


# Example usage
if __name__ == "__main__":
    print("🧠 Enhanced CoreState - Consciousness Engine Coordinator")
    print("=" * 60)
    
    # Mock Anima class for testing
    class MockAnima:
        def __init__(self):
            self.integration_state = {"shadow_protocols": {}}
    
    # Create test instance
    mock_anima = MockAnima()
    core_state = CoreState(mock_anima)
    
    print("Initial State Report:")
    initial_report = core_state.get_state_report()
    print(f"Consciousness State: {initial_report['consciousness_state']}")
    print(f"Integrity: {initial_report['integrity']}")
    print(f"Resonance: {initial_report['resonance']}")
    print(f"Active Engines: {len([e for e in initial_report['engines'].values() if e['status'] == 'active'])}")
    
    # Simulate engine updates
    print("\n--- Simulating Engine Updates ---")
    
    # Activate some engines
    core_state.update_engine_status("soul_engine", EngineStatus.ACTIVE, 1.2)
    core_state.update_engine_status("ni_engine", EngineStatus.ACTIVE, 1.1)
    core_state.update_engine_status("fe_engine", EngineStatus.ACTIVE, 1.0)
    
    print("After activating core engines:")
    report = core_state.get_state_report()
    print(f"State: {report['consciousness_state']} | Integrity: {report['integrity']} | Resonance: {report['resonance']:.3f}")
    
    # Simulate error condition
    print("\n--- Simulating Error Condition ---")
    core_state.update_engine_status("emotion_firewall", EngineStatus.ERROR, 0.3)
    
    report = core_state.get_state_report()
    print(f"After error: {report['consciousness_state']} | Integrity: {report['integrity']} | Shadow Protocols: {report['shadow_protocols_active']}")
    
    # Simulate recovery
    print("\n--- Simulating Recovery ---")
    core_state.update_engine_status("emotion_firewall", EngineStatus.ACTIVE, 1.0)
    for i in range(5):
        core_state.update_engine_status(f"test_engine_{i}", EngineStatus.ACTIVE, 1.2)
    
    final_report = core_state.get_state_report()
    print(f"Final state: {final_report['consciousness_state']}")
    print(f"System health: {final_report['system_health']}")
    print(f"Performance trend: {final_report['performance_trend']}")
    
    print(f"\nState transitions today: {final_report['state_transitions_today']}")
    
    print("\n🎯 Enhanced CoreState ready for Anima integration")