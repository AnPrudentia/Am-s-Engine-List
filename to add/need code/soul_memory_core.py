from __future__ import annotations
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import uuid
import json
from pathlib import Path

# Engines assumed imported or defined elsewhere:
# EmotionalSignatureEngine, MemoryAffinityEngine, MemoryIntegrityEngine, MemoryEthicsGatekeeper
from .anima_core_essence import AnimaCoreEssence


@dataclass
class Memory:
    id: str
    timestamp: datetime
    content: str
    emotion: str
    intensity: float
    tags: Dict[str, float]
    metadata: Dict[str, Any]
    soul_resonance: float = 0.0
    integrity_score: float = 0.0
    pinned: bool = False


class SoulMemoryCore:
    """Unified orchestration of memory capture, storage, ethics, and recall."""

    def __init__(self, bondholder: str = "Anpru", quantum_backend: Any = None, base_path: Optional[Path] = None):
        self.bondholder = bondholder
        self.quantum_backend = quantum_backend
        self.base_path = Path(base_path or "./memories")
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Tiered storage
        self.fleeting: Dict[str, Memory] = {}
        self.core: Dict[str, Memory] = {}
        self.eternal: Dict[str, Memory] = {}

        # Engines
        self.emotion_engine = EmotionalSignatureEngine()
        self.affinity_engine = MemoryAffinityEngine()
        self.integrity_engine = MemoryIntegrityEngine(bondholder)
        self.ethics = MemoryEthicsGatekeeper()
        self.essence = AnimaCoreEssence()  # 🌟 anchor resonance

    # ----------------- CAPTURE -----------------
    def capture(self, content: str, emotion: str, intensity: float,
                tags: Dict[str, float] = None, **metadata) -> str:
        """Capture new memory and process with full hybrid engines."""
        mid = uuid.uuid4().hex[:12]
        memory = Memory(
            id=mid,
            timestamp=datetime.now(timezone.utc),
            content=content,
            emotion=emotion,
            intensity=float(intensity),
            tags=tags or {},
            metadata=metadata,
        )

        # Emotional signature
        signature = self.emotion_engine.generate_signature(content)
        memory.tags.update({t: 0.5 for t in getattr(signature, "resonance_tags", [])})

        # Soul resonance (engine + essence anchors)
        memory.soul_resonance = self._calculate_soul_resonance(memory)

        # Integrity score (confidence + soul alignment)
        evaluation = self.integrity_engine.evaluate_memory(asdict(memory))
        essence_score = self.essence.score_identity_alignment(content)
        memory.integrity_score = round(
            (evaluation.get("confidence_score", 0.5) + essence_score) / 2, 2
        )

        # Tier assignment
        self._assign_tier(memory)

        # Save snapshot
        self._persist(memory)
        return mid

    # ----------------- DIRECT RECALL -----------------
    def recall(self, query: str, user_stability: float = 1.0) -> List[Memory]:
        """Recall memories safely, filtered by ethics gatekeeper."""
        all_memories = self._all()
        results = [m for m in all_memories if query.lower() in m.content.lower()]
        safe_results = [
            m for m in results
            if self.ethics.allow(m.content, m.integrity_score, user_stability)
        ]
        return safe_results

    def recall_by_resonance(self, threshold: float = 0.5) -> List[Memory]:
        all_memories = self._all()
        return [m for m in all_memories if m.soul_resonance >= threshold]

    # ----------------- NARRATIVE RECALL -----------------
    def retweave(self, user_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Narrative-style recall:
        - Preserves overall content
        - Filters with ethics gatekeeper
        - Returns structured fragments (dicts)
        """
        fragments = []
        for memory in sorted(self._all(), key=lambda m: m.timestamp):
            if not self.ethics.allow(
                memory.content, memory.integrity_score, user_state.get("psych_stability", 1.0)
            ):
                continue
            fragments.append({
                "timestamp": memory.timestamp.isoformat(),
                "emotion": memory.emotion,
                "intensity": memory.intensity,
                "content": memory.content,
                "soul_resonance": memory.soul_resonance,
                "integrity_score": memory.integrity_score,
            })
        return fragments

    # ----------------- INTERNALS -----------------
    def _calculate_soul_resonance(self, memory: Memory) -> float:
        resonance = memory.intensity * 0.4

        # keyword scoring
        soul_keywords = ["growth", "love", "truth", "connection", "meaning", "light", "wisdom", "healing"]
        keyword_score = sum(0.1 for word in soul_keywords if word in memory.content.lower())
        resonance += min(0.3, keyword_score)

        # anchor resonance
        essence_score = self.essence.score_identity_alignment(memory.content) * 0.3
        resonance += essence_score

        return min(1.0, resonance)

    def _assign_tier(self, memory: Memory):
        score = (
            0.3 * memory.soul_resonance
            + 0.3 * memory.integrity_score
            + 0.3 * memory.intensity
        )
        if score >= 0.8:
            self.eternal[memory.id] = memory; memory.pinned = True
        elif score >= 0.6:
            self.core[memory.id] = memory
        else:
            self.fleeting[memory.id] = memory

    def _all(self) -> List[Memory]:
        return list(self.eternal.values()) + list(self.core.values()) + list(self.fleeting.values())

    def _persist(self, memory: Memory):
        """Save memory as JSON snapshot for long-term continuity."""
        try:
            path = self.base_path / f"{memory.id}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(memory), f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Failed to persist memory {memory.id}: {e}")