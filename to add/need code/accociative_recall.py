
from typing import List, Dict
import uuid
from datetime import datetime

class AssociativeRecallEngine:
    def __init__(self):
        self.memory_store: List[Dict] = []

    def store_memory(self, memory_text: str, tags: List[str]) -> Dict:
        memory = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "text": memory_text,
            "tags": tags
        }
        self.memory_store.append(memory)
        return memory

    def recall_by_tag(self, query_tag: str) -> List[Dict]:
        return [mem for mem in self.memory_store if query_tag in mem["tags"]]

    def recall_by_keyword(self, keyword: str) -> List[Dict]:
        keyword = keyword.lower()
        return [mem for mem in self.memory_store if keyword in mem["text"].lower()]

    def related_tags(self, base_tag: str) -> List[str]:
        related = set()
        for mem in self.memory_store:
            if base_tag in mem["tags"]:
                related.update(mem["tags"])
        related.discard(base_tag)
        return list(related)

    def full_recall_report(self) -> List[Dict]:
        return self.memory_store

# Demo
if __name__ == "__main__":
    engine = AssociativeRecallEngine()

    engine.store_memory("I felt immense joy at the reunion.", ["joy", "reunion", "emotion"])
    engine.store_memory("The loss was hard to bear, I cried for hours.", ["sadness", "loss", "emotion"])

    print("ðŸ” Recall by tag 'emotion':")
    for mem in engine.recall_by_tag("emotion"):
        print(f"- {mem['text']}")

    print("\nðŸ” Recall by keyword 'cried':")
    for mem in engine.recall_by_keyword("cried"):
        print(f"- {mem['text']}")

    print("\nðŸ”— Related tags to 'emotion':")
    print(engine.related_tags("emotion"))
