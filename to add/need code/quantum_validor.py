import json
import logging
import pkgutil
import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Configure root logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Validator Rule Base
# ------------------------------
class ValidatorRule(ABC):
    level: str = "info"

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get(self.__class__.__name__, {})

    @abstractmethod
    def validate(self, circuit, backend) -> List[str]:
        ...


# ------------------------------
# Built-in Rules
# ------------------------------

class QubitCountRule(ValidatorRule):
    level = "error"

    def validate(self, circuit, backend) -> List[str]:
        msgs = []
        max_q = backend.configuration().n_qubits
        used = circuit.num_qubits
        if used > max_q:
            msgs.append(f"Circuit uses {used} qubits; hardware supports only {max_q}.")
        return msgs


class DepthRule(ValidatorRule):
    level = "warning"

    def validate(self, circuit, backend) -> List[str]:
        msgs = []
        depth = circuit.depth()
        max_depth = self.config.get("max_depth", 100)
        if depth > max_depth:
            msgs.append(
                f"Circuit depth {depth} exceeds threshold {max_depth}; "
                "may exceed decoherence window."
            )
        return msgs


class CXGateCountRule(ValidatorRule):
    level = "warning"

    def validate(self, circuit, backend) -> List[str]:
        msgs = []
        max_cx = self.config.get("max_cx_count", 50)
        cx_ops = sum(1 for op, _ in circuit.data if op.name.lower() in ("cx", "cnot"))
        if cx_ops > max_cx:
            msgs.append(
                f"Found {cx_ops} two-qubit gates (CX/CNOT); recommended max is {max_cx}."
            )
        return msgs


class DisconnectedQubitRule(ValidatorRule):
    level = "info"

    def validate(self, circuit, backend) -> List[str]:
        used_qubits = set()
        for instr, qargs, _ in circuit.data:
            for qubit in qargs:
                used_qubits.add(qubit.index)

        total = set(range(circuit.num_qubits))
        unused = sorted(total - used_qubits)
        if unused:
            return [f"Qubits unused in circuit: {unused}"]
        return []


# ------------------------------
# Quantum Validator
# ------------------------------

class QuantumBuildValidator:
    def __init__(
        self,
        circuit: Any,
        backend: Any,
        config_path: Optional[str] = None,
        rules_package: Optional[str] = None,
    ):
        self.circuit = circuit
        self.backend = backend
        self.report = {"errors": [], "warnings": [], "info": []}
        self.config = self._load_config(config_path)
        self.rules = self._discover_rules(rules_package)

    def _load_config(self, path: Optional[str]) -> Dict[str, Any]:
        default = {
            "DepthRule": {"max_depth": 100},
            "CXGateCountRule": {"max_cx_count": 50},
            "QubitCountRule": {"enabled": True},
            "DisconnectedQubitRule": {"enabled": True}
        }
        if not path:
            return default
        try:
            with open(path) as f:
                user_cfg = json.load(f)
            default.update(user_cfg)
            return default
        except Exception as e:
            logger.warning(f"Failed to load config '{path}': {e}. Using defaults.")
            return default

    def _discover_rules(self, package: Optional[str]) -> List[ValidatorRule]:
        rules: List[ValidatorRule] = []

        for cls in ValidatorRule.__subclasses__():
            rule_cfg = self.config.get(cls.__name__, {})
            if rule_cfg.get("enabled", True):
                rules.append(cls(self.config))

        if package:
            for finder, name, _ in pkgutil.iter_modules([package.replace(".", "/")]):
                module = importlib.import_module(f"{package}.{name}")
                for attr in dir(module):
                    obj = getattr(module, attr)
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, ValidatorRule)
                        and obj is not ValidatorRule
                    ):
                        rule_cfg = self.config.get(obj.__name__, {})
                        if rule_cfg.get("enabled", True):
                            rules.append(obj(self.config))

        logger.info(f"Discovered {len(rules)} validation rules.")
        return rules

    def run(self) -> Dict[str, List[str]]:
        for rule in self.rules:
            try:
                messages = rule.validate(self.circuit, self.backend)
                for msg in messages:
                    self.report[f"{rule.level}s"].append(f"{rule.__class__.__name__}: {msg}")
                    logger.debug(f"[{rule.level.upper()}] {msg}")
            except Exception as e:
                logger.error(f"Rule {rule.__class__.__name__} failed: {e}")

        if not any(self.report.values()):
            self.report["info"].append("No critical issues detected.")
        return self.report


# ------------------------------
# Export Report (Text)
# ------------------------------

def export_report(report: Dict[str, List[str]], filename: str = "validation_report.txt"):
    with open(filename, "w") as f:
        for level in ("errors", "warnings", "info"):
            f.write(f"\n{level.upper()}:\n")
            for line in report[level]:
                f.write(f"  - {line}\n")


# ------------------------------
# Example Usage
# ------------------------------

if __name__ == "__main__":
    from qiskit import QuantumCircuit, Aer

    qc = QuantumCircuit(10)
    for i in range(9):
        qc.cx(i, i + 1)
    qc.h(range(10))  # Apply H gate to all qubits

    backend = Aer.get_backend("qasm_simulator")

    validator = QuantumBuildValidator(qc, backend, config_path=None)
    result = validator.run()

    print("\nValidation Report:")
    for level in ("errors", "warnings", "info"):
        print(f"\n{level.upper()}:")
        for line in result[level]:
            print("  -", line)

    # Optional: Save to file
    export_report(result, "validation_report.txt")
    print("\nReport exported to validation_report.txt")