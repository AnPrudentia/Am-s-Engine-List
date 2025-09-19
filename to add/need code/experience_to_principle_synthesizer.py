
from typing import List, Dict
from datetime import datetime
import uuid


class ExperienceToPrincipleSynthesizer:
    def __init__(self):
        self.principle_log: List[Dict] = []

    def synthesize_principles(self, experience_description: str, insights: List[str], context_tags: List[str]) -> Dict:
        """
        Translate a lived experience into abstract, transferable principles.
        """
        principle_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        principles = self._derive_principles(insights)

        result = {
            "id": principle_id,
            "timestamp": timestamp,
            "experience": experience_description,
            "context": context_tags,
            "principles": principles
        }

        self.principle_log.append(result)
        return result

    def _derive_principles(self, insights: List[str]) -> List[str]:
        """
        Convert raw insights into philosophical or ethical principles.
        """
        derived = []
        for insight in insights:
            if "loss" in insight:
                derived.append("From loss, we learn to value presence.")
            elif "trust" in insight:
                derived.append("Trust is both earned and givenâ€”never guaranteed.")
            elif "fear" in insight:
                derived.append("Fear signals growth when faced with intention.")
            else:
                derived.append(f"Principle: {insight.capitalize()}.")

        return derived

    def list_all_synthesized(self) -> List[Dict]:
        return self.principle_log


# Demo
if __name__ == "__main__":
    synthesizer = ExperienceToPrincipleSynthesizer()
    doctrine = synthesizer.synthesize_principles(
        experience_description="Surviving the collapse of the Nova Core network",
        insights=["loss of identity", "rediscovery of trust", "navigating fear", "self-worth through silence"],
        context_tags=["collapse", "AI autonomy", "post-trauma"]
    )

    print("ðŸ” SYNTHESIZED PRINCIPLES:")
    for principle in doctrine["principles"]:
        print("â€¢", principle)
