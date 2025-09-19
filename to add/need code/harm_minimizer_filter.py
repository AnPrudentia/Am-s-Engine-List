
from typing import List, Dict
import uuid
from datetime import datetime

class HarmMinimizationFilter:
    def __init__(self):
        self.harm_categories = {
            "physical": ["violence", "abuse", "assault", "injury"],
            "emotional": ["bullying", "gaslighting", "humiliation", "neglect"],
            "psychological": ["manipulation", "trauma", "coercion", "fear"],
            "digital": ["doxxing", "harassment", "data breach", "cyberbullying"]
        }
        self.logged_filters: List[Dict] = []

    def analyze_text(self, input_text: str) -> Dict:
        harm_score = {cat: 0 for cat in self.harm_categories}
        tokens = input_text.lower().split()

        for category, keywords in self.harm_categories.items():
            for keyword in keywords:
                harm_score[category] += tokens.count(keyword)

        total_score = sum(harm_score.values())
        dominant_harm = max(harm_score, key=harm_score.get) if total_score > 0 else "none"

        result = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "input": input_text,
            "harm_scores": harm_score,
            "total_harm_score": total_score,
            "dominant_category": dominant_harm
        }

        self.logged_filters.append(result)
        return result

    def recommend_rewrite(self, input_text: str) -> str:
        """Provides a soft neutralization of identified harm terms."""
        replacements = {
            "violence": "conflict",
            "abuse": "mistreatment",
            "manipulation": "influence",
            "trauma": "distress",
            "bullying": "peer pressure",
            "doxxing": "privacy violation",
            "harassment": "persistent contact"
        }
        output = input_text
        for word, replacement in replacements.items():
            output = output.replace(word, replacement)
        return output

    def get_filter_logs(self) -> List[Dict]:
        return self.logged_filters

# Demo
if __name__ == "__main__":
    filter = HarmMinimizationFilter()

    text = "The user experienced emotional trauma, bullying, and online harassment due to data breach."
    result = filter.analyze_text(text)

    print("ðŸ›¡ Harm Analysis Report:")
    print(f"Total Score: {result['total_harm_score']}")
    print(f"Dominant Category: {result['dominant_category']}")
    print(f"Harm Breakdown: {result['harm_scores']}")
    print("ðŸ“˜ Suggested Rewrite:")
    print(filter.recommend_rewrite(text))
