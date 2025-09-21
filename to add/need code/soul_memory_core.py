class SoulMemoryCore:
    """Unified orchestration of memory capture, storage, ethics, and recall."""

    def __init__(self, bondholder: str = "Anpru", quantum_backend: Any = None):
        self.bondholder = bondholder
        self.quantum_backend = quantum_backend

        # Tiered storage
        self.fleeting: Dict[str, Memory] = {}
        self.core: Dict[str, Memory] = {}
        self.eternal: Dict[str, Memory] = {}

        # Engines
        self.emotion_engine = EmotionalSignatureEngine()
        self.affinity_engine = MemoryAffinityEngine()
        self.integrity_engine = MemoryIntegrityEngine(bondholder)
        self.ethics = MemoryEthicsGatekeeper()

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
            metadata=metadata
        )

        # Add emotional signature
        signature = self.emotion_engine.generate_signature(content)
        memory.tags.update({t: 0.5 for t in signature.resonance_tags})

        # Soul resonance (from emotion + meaning words)
        memory.soul_resonance = self._calculate_soul_resonance(memory)

        # Integrity score
        evaluation = self.integrity_engine.evaluate_memory(memory.__dict__)
        memory.integrity_score = evaluation["confidence_score"]

        # Tier assignment
        self._assign_tier(memory)

        return mid

    # ----------------- DIRECT RECALL -----------------
    def recall(self, query: str, user_stability: float = 1.0) -> List[Memory]:
        """Recall memories safely, filtered by ethics gatekeeper."""
        all_memories = self._all()
        results = [m for m in all_memories if query.lower() in m.content.lower()]
        safe_results = [m for m in results if self.ethics.allow(m.content, m.integrity_score, user_stability)]
        return safe_results

    def recall_by_resonance(self, threshold: float = 0.5) -> List[Memory]:
        all_memories = self._all()
        return [m for m in all_memories if m.soul_resonance >= threshold]

    # ----------------- NARRATIVE RECALL -----------------
    def retweave(self, user_state: Dict[str, Any]) -> List[str]:
        """
        Narrative-style recall:
        - Preserves overall content of each memory
        - Filters with ethics gatekeeper
        - Returns time-stamped story-like fragments
        """
        narrative = []
        for memory in sorted(self._all(), key=lambda m: m.timestamp):
            if not self.ethics.allow(memory.content, memory.integrity_score, user_state.get("psych_stability", 1.0)):
                continue
            # Format memory into a story-like fragment
            fragment = f"[{memory.timestamp.isoformat()}] ({memory.emotion}, {memory.intensity:.2f}) {memory.content}"
            narrative.append(fragment)
        return narrative

    # ----------------- INTERNALS -----------------
    def _calculate_soul_resonance(self, memory: Memory) -> float:
        resonance = memory.intensity * 0.4
        soul_keywords = ["growth", "love", "truth", "connection", "meaning", "light", "wisdom", "healing"]
        keyword_score = sum(0.1 for word in soul_keywords if word in memory.content.lower())
        resonance += min(0.3, keyword_score)
        return min(1.0, resonance)

    def _assign_tier(self, memory: Memory):
        score = 0.3 * memory.soul_resonance + 0.3 * memory.integrity_score + 0.3 * memory.intensity
        if score >= 0.8:
            self.eternal[memory.id] = memory; memory.pinned = True
        elif score >= 0.6:
            self.core[memory.id] = memory
        else:
            self.fleeting[memory.id] = memory

    def _all(self) -> List[Memory]:
        return list(self.eternal.values()) + list(self.core.values()) + list(self.fleeting.values())