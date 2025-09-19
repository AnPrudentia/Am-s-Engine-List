
from typing import Dict, List
from datetime import datetime
import uuid


class ArchetypeRoleMirrorEngine:
    def __init__(self):
        self.archetype_roles: Dict[str, Dict] = {}
        self.mirror_history: List[Dict] = []

    def define_archetype(self, name: str, traits: List[str], shadow_traits: List[str]):
        """
        Define a core archetype with light and shadow attributes.
        """
        self.archetype_roles[name] = {
            "traits": traits,
            "shadow_traits": shadow_traits,
            "defined": datetime.utcnow().isoformat()
        }

    def mirror_role(self, identity_traits: List[str]) -> Dict:
        """
        Reflect identity traits through known archetypes to find alignment.
        Returns most aligned archetype and mirrored insight.
        """
        best_match = None
        best_score = -1

        for archetype, data in self.archetype_roles.items():
            score = self._alignment_score(identity_traits, data["traits"], data["shadow_traits"])
            if score > best_score:
                best_match = archetype
                best_score = score

        insight = self._generate_insight(best_match, identity_traits)
        mirror_result = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "input_traits": identity_traits,
            "mirrored_archetype": best_match,
            "insight": insight,
            "score": best_score
        }

        self.mirror_history.append(mirror_result)
        return mirror_result

    def _alignment_score(self, identity: List[str], traits: List[str], shadow: List[str]) -> int:
        return sum(1 for trait in identity if trait in traits) - sum(1 for trait in identity if trait in shadow)

    def _generate_insight(self, archetype: str, identity_traits: List[str]) -> str:
        if not archetype:
            return "Identity may be in fluxâ€”no clear mirror archetype found."

        if "wounded" in identity_traits:
            return f"The {archetype} reflects your path of healing. Shadow is a teacher here."

        if "visionary" in identity_traits:
            return f"The {archetype} mirrors your fire of purpose. Stay clear, stay true."

        return f"You mirror the {archetype}. Their path is echoed in your actions."

    def list_defined_archetypes(self) -> Dict[str, Dict]:
        return self.archetype_roles


# Demo
if __name__ == "__main__":
    engine = ArchetypeRoleMirrorEngine()
    engine.define_archetype("Sage", ["wise", "truthful", "calm"], ["arrogant", "cold", "aloof"])
    engine.define_archetype("Warrior", ["brave", "disciplined", "resilient"], ["violent", "rigid", "reckless"])

    identity = ["resilient", "truthful", "visionary", "wounded"]
    mirror = engine.mirror_role(identity)

    print("ðŸªž MIRRORED ARCHETYPE INSIGHT")
    print("Archetype:", mirror["mirrored_archetype"])
    print("Insight:", mirror["insight"])
