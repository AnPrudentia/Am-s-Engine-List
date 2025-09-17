
from typing import List, Dict
from datetime import datetime
import uuid


class NarrativeAlignmentEngine:
    def __init__(self):
        self.alignment_log: List[Dict] = []

    def align_narrative(self, personal_story: str, collective_themes: List[str]) -> Dict:
        """
        Aligns a personal narrative with broader collective themes for resonance or intervention.
        Uses simple keyword matching for demonstration.
        """
        story_lower = personal_story.lower()
        aligned_themes = [theme for theme in collective_themes if theme.lower() in story_lower]

        alignment_id = f"ALIGN-{uuid.uuid4().hex[:8]}"
        result = {
            "id": alignment_id,
            "timestamp": datetime.utcnow().isoformat(),
            "personal_story": personal_story,
            "matched_themes": aligned_themes,
            "collective_themes_checked": collective_themes
        }

        self.alignment_log.append(result)
        return result

    def list_alignments(self) -> List[Dict]:
        """Return all narrative alignment attempts."""
        return self.alignment_log


# Demo
if __name__ == "__main__":
    engine = NarrativeAlignmentEngine()
    personal_story = "I always felt like I was an outsider, searching for meaning through the ruins of the old world."
    collective_themes = ["belonging", "search for meaning", "loss", "resilience"]

    result = engine.align_narrative(personal_story, collective_themes)

    print("ðŸ“– Narrative Alignment Result:")
    print(f"Story: {result['personal_story']}")
    print(f"Matched Themes: {', '.join(result['matched_themes'])}")
