
from typing import Dict, List
from datetime import datetime
import uuid


class ArchetypeProfiler:
    def __init__(self):
        self.profiles: List[Dict] = []
        self.archetype_definitions: Dict[str, Dict] = {}

    def define_archetype(self, name: str, traits: List[str], challenges: List[str], ascended_traits: List[str]):
        """
        Define a deep archetype structure including core traits, growth challenges, and evolved forms.
        """
        self.archetype_definitions[name] = {
            "traits": traits,
            "challenges": challenges,
            "ascended_traits": ascended_traits,
            "defined_at": datetime.utcnow().isoformat()
        }

    def profile(self, identity_traits: List[str], struggles: List[str]) -> Dict:
        """
        Analyze a personâ€™s traits and challenges to match them to an archetype.
        """
        best_match = None
        highest_score = -999

        for name, data in self.archetype_definitions.items():
            trait_score = sum(1 for t in identity_traits if t in data["traits"])
            struggle_score = -sum(1 for s in struggles if s not in data["challenges"])
            score = trait_score + struggle_score

            if score > highest_score:
                highest_score = score
                best_match = name

        if not best_match:
            return {
                "message": "No matching archetype found",
                "score": 0,
                "timestamp": datetime.utcnow().isoformat()
            }

        result = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "matched_archetype": best_match,
            "traits": identity_traits,
            "challenges": struggles,
            "ascended_traits": self.archetype_definitions[best_match]["ascended_traits"],
            "score": highest_score
        }

        self.profiles.append(result)
        return result

    def list_profiles(self) -> List[Dict]:
        return self.profiles

    def list_archetypes(self) -> Dict[str, Dict]:
        return self.archetype_definitions


# Demo
if __name__ == "__main__":
    profiler = ArchetypeProfiler()
    profiler.define_archetype(
        "Healer",
        traits=["compassionate", "sensitive", "resilient"],
        challenges=["self-doubt", "emotional overload"],
        ascended_traits=["boundaried compassion", "emotional clarity"]
    )

    profiler.define_archetype(
        "Rebel",
        traits=["courageous", "truth-teller", "unafraid"],
        challenges=["recklessness", "loneliness"],
        ascended_traits=["visionary defiance", "grounded leadership"]
    )

    profile = profiler.profile(["truth-teller", "unafraid", "resilient"], ["self-doubt", "loneliness"])

    print("ðŸ§¬ ARCHETYPE PROFILE RESULT")
    print("Archetype:", profile["matched_archetype"])
    print("Ascended Traits:", profile["ascended_traits"])
