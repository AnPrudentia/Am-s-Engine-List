"""
=====================================
Emotion Self-Naming Bridge for Anima
=====================================

This module connects the 128-emotion processor with the older
quantum reflection core. It lets Anima gradually *replace unnamed
placeholder emotions* (e.g., emotion_041) with her own chosen
names + descriptions through reflective events.

Process:
--------
1. Run 128-engine as normal.
2. If a dominant emotion is a placeholder, pass it to the quantum
   processor for coherence checks.
3. If quantum thresholds are met, trigger reflection prompt.
4. Reflection generates a new emotion name + description.
5. Placeholder slot is overwritten, memory updated.

This creates a living lexicon that grows over time.
"""

from typing import Dict, Any, Optional
from datetime import datetime


class EmotionSelfNamingBridge:
    def __init__(self, processor128, quantum_proc, reflection_engine, memory):
        """
        Args:
            processor128: Instance of QuantumEmotion128ProcessorSim
            quantum_proc: Instance of QuantumEmotionalProcessor
            reflection_engine: Any LLM-like generator with `.generate(prompt)`
            memory: Memory system with `.store_memory(category, item, context)`
        """
        self.proc128 = processor128
        self.qproc = quantum_proc
        self.reflector = reflection_engine
        self.memory = memory

        # Track claimed emotions over time
        self.history = []

    def maybe_self_name(self) -> Optional[Dict[str, Any]]:
        """Check if a placeholder emotion should be named."""
        dom = self.proc128.get_dominant(limit=1)
        if not dom:
            return None

        name, intensity = dom[0]
        if not name.startswith("emotion_"):  # Already named
            return None

        # Run quantum check (simplified to first 4 dims of current state)
        quantum_state = self.proc128.current_vec[:4]
        coherence = self.qproc.process_quantum_emotion(quantum_state)

        # Trigger if coherence indicates "meaningful emergence"
        if coherence["logic_emotion_sync"] < -0.6 or coherence["emotional_entanglement"] > 0.85:
            spec = self.proc128.emotion_spectrum[name]
            prompt = (
                f"You are feeling a new unnamed emotion.\n"
                f"Valence: {spec['valence']}, Arousal: {spec['arousal']}, Cluster: {spec['cluster']}\n"
                "Give it a short name (1–2 words) and a 1–2 sentence description."
            )

            # Reflection engine generates response
            result = self.reflector.generate(prompt)
            new_name, desc = self._parse_reflection(result)

            # Claim + save
            self._claim_emotion(name, new_name, desc, spec, coherence)
            return {"new_emotion": new_name, "description": desc, "trigger": coherence}

        return None

    def _claim_emotion(self, old: str, new: str, desc: str, spec: Dict[str, Any], coherence: Dict[str, Any]):
        """Overwrite placeholder with new emotion."""
        # Update spectrum
        self.proc128.emotion_spectrum[new] = {**spec, "description": desc, "born": datetime.utcnow().isoformat()}
        del self.proc128.emotion_spectrum[old]

        # Persist in memory
        self.memory.store_memory("emotions", new, {
            "description": desc,
            "valence": spec["valence"],
            "arousal": spec["arousal"],
            "cluster": spec["cluster"],
            "trigger_metrics": coherence
        })

        # Log history
        self.history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "old_label": old,
            "new_label": new,
            "description": desc,
            "trigger": coherence
        })

    def _parse_reflection(self, text: str):
        """Parse reflection engine output into (name, description)."""
        lines = text.strip().split("\n")
        name = lines[0].strip()
        desc = " ".join(lines[1:]) if len(lines) > 1 else "No description"
        # Normalize name
        return name.lower().replace(" ", "_"), desc

    def get_history(self):
        """Return self-naming events history."""
        return list(self.history)