from typing import Dict, List
from datetime import datetime
import uuid


class AdaptivePersonaShaper:
    def __init__(self):
        self.shaped_personas: List[Dict] = []

    def shape_persona(self, context: str, goals: List[str], emotional_state: str, environment_factors: List[str]) -> Dict:
        persona_id = f"APS-{uuid.uuid4().hex[:8]}"
        persona_profile = {
            "id": persona_id,
            "timestamp": datetime.utcnow().isoformat(),
            "context": context,
            "goals": goals,
            "emotional_state": emotional_state,
            "environment_factors": environment_factors,
            "generated_persona": self._generate_profile(context, goals, emotional_state, environment_factors)
        }
        self.shaped_personas.append(persona_profile)
        return persona_profile

    def _generate_profile(self, context: str, goals: List[str], emotional_state: str, environment_factors: List[str]) -> str:
        shape = f"{context.upper()}-{emotional_state.upper()}-{'-'.join([g[:3].upper() for g in goals])}-ENV:{len(environment_factors)}"
        return shape

    def list_all_shaped_personas(self) -> List[Dict]:
        return self.shaped_personas
