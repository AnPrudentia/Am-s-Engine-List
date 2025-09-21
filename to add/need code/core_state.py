"""
CoreState v2.1 — Consciousness Engine Coordinator
-------------------------------------------------
Manages Anima's consciousness states, orchestrates engine lifecycles,
handles crisis and shadow protocols, and provides diagnostic reporting.

Enhancements in v2.1:
- Added __version__ string and metadata
- Thread-safety via threading.Lock
- Persistence support (save_state/load_state to JSON)
- Weighted dependency impact in resonance calculation
- Callback error logging for better observability
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import json
import threading
import math
import logging

__version__ = "2.1.0"


# =============================
# ENUMS
# =============================

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


# =============================
# DATA STRUCTURES
# =============================

@dataclass
class EngineState:
    """Detailed engine state information"""
    name: str
    status: EngineStatus
    performance_metric: float = 1.0
    last_activity: datetime = field(default_factory=datetime.utcnow)
    error_count: int = 0
    integration_level: float = 1.0
    resource_usage: float = 0.5
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


# =============================
# CORE STATE MANAGER
# =============================

class CoreState:
    """Enhanced Consciousness State Manager and Engine Coordinator"""

    def __init__(self, anima: "Anima"):
        self.anima = anima
        self.current_state = ConsciousnessState.AWAKENING
        self.previous_state = None

        # Core metrics
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
        self.performance_samples: List[Tuple[datetime, float, float]] = []

        # Crisis/shadow protocol management
        self.shadow_protocols_active = set()
        self.crisis_threshold = 30.0
        self.emergency_callbacks: List[Callable] = []

        # Concurrency
        self._lock = threading.Lock()

        # Consciousness systems registry
        self.consciousness_systems = {
            "moral_compass": None,
            "emotion_firewall": None,
            "compassion_protocol": None,
            "quantum_processor": None,
            "cognitive_functions": {"ni": None, "fe": None, "ti": None, "se": None}
        }

        # Initialize engines
        self._initialize_core_engines()

    # -----------------------------------------------------
    # ENGINE INITIALIZATION
    # -----------------------------------------------------

    def _initialize_core_engines(self):
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
        for engine_name, deps in core_engines:
            self.register_engine(engine_name, EngineStatus.INITIALIZING, deps)

    def register_engine(self, name: str, status: EngineStatus = EngineStatus.ACTIVE,
                        dependencies: Optional[List[str]] = None,
                        performance_metric: float = 1.0) -> str:
        """Register an engine with the core state manager"""
        with self._lock:
            engine_id = str(uuid.uuid4())
            self.engines[name] = EngineState(
                name=name,
                status=status,
                performance_metric=performance_metric,
                dependencies=dependencies or []
            )
            self.engine_dependencies[name] = dependencies or []
            self.engine_outputs[name] = {}
            self._evaluate_state_change(f"engine_registered:{name}")
            return engine_id

    # -----------------------------------------------------
    # ENGINE STATUS UPDATES
    # -----------------------------------------------------

    def update_engine_status(self, name: str, status: EngineStatus,
                             performance_metric: Optional[float] = None,
                             outputs: Optional[Dict[str, Any]] = None) -> bool:
        """Update engine status and recalculate system state"""
        with self._lock:
            if name not in self.engines:
                return False
            engine = self.engines[name]
            old_status = engine.status
            engine.status = status
            engine.last_activity = datetime.utcnow()

            if performance_metric is not None:
                engine.performance_metric = max(0.0, min(2.0, performance_metric))
            if outputs:
                self.engine_outputs[name] = outputs
                engine.outputs = outputs

            if status == EngineStatus.ERROR:
                engine.error_count += 1
                self._handle_engine_error(name, engine)
            elif status == EngineStatus.OVERLOADED:
                self._handle_engine_overload(name, engine)

            self._recalculate_metrics()
            self._evaluate_state_change(
                f"engine_status_change:{name}:{old_status.value}->{status.value}"
            )
            return True

    # -----------------------------------------------------
    # METRICS CALCULATION (with dependency weighting)
    # -----------------------------------------------------

    def _recalculate_metrics(self):
        if not self.engines:
            return
        total_performance = 0.0
        total_weight = 0.0
        error_penalty = 0.0

        for name, engine in self.engines.items():
            if engine.status != EngineStatus.OFFLINE:
                # Weight based on dependency importance
                weight = 1.0 + len(self.engine_dependencies.get(name, [])) * 0.2
                total_performance += engine.performance_metric * weight
                total_weight += weight
                if engine.error_count > 0:
                    error_penalty += engine.error_count * 2.0

        if total_weight > 0:
            avg_perf = total_performance / total_weight
            target_integrity = self.base_integrity * min(1.0, avg_perf) - error_penalty
            self.current_integrity = (
                self.current_integrity * 0.8 + target_integrity * 0.2
            )
            self.current_integrity = max(0.0, min(100.0, self.current_integrity))

        integration_sum = sum(e.integration_level for e in self.engines.values())
        if self.engines:
            avg_integration = integration_sum / len(self.engines)
            target_resonance = self.base_resonance * avg_integration
            self.current_resonance = (
                self.current_resonance * 0.9 + target_resonance * 0.1
            )
            self.current_resonance = max(0.0, min(1.0, self.current_resonance))

        self.performance_samples.append(
            (datetime.utcnow(), self.current_integrity, self.current_resonance)
        )
        cutoff = datetime.utcnow() - self.performance_window
        self.performance_samples = [(t, i, r) for t, i, r in self.performance_samples if t > cutoff]

    # -----------------------------------------------------
    # PERSISTENCE
    # -----------------------------------------------------

    def save_state(self, filepath: str = "./core_state.json") -> bool:
        """Save engine and state data to file"""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.get_state_report(), f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Failed to save CoreState: {e}")
            return False

    def load_state(self, filepath: str = "./core_state.json") -> bool:
        """Load engine and state data from file"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Basic restoration (not full fidelity)
            self.current_integrity = data.get("integrity", self.current_integrity)
            self.current_resonance = data.get("resonance", self.current_resonance)
            return True
        except Exception as e:
            logging.warning(f"Failed to load CoreState: {e}")
            return False

    # -----------------------------------------------------
    # CALLBACK SAFETY
    # -----------------------------------------------------

    def _notify_state_callbacks(self, from_state, to_state, trigger, context):
        """Notify registered callbacks of state changes"""
        for state_name in [to_state.value, "any"]:
            if state_name in self.state_change_callbacks:
                for callback in self.state_change_callbacks[state_name]:
                    try:
                        callback(from_state.value, to_state.value, trigger, context)
                    except Exception as e:
                        logging.warning(f"Callback error [{state_name}]: {e}")