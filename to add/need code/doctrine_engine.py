from typing import Dict, List
from datetime import datetime
import uuid

class DoctrineEngine:
    def __init__(self):
        self.doctrines: List[Dict] = []

    def create_doctrine(self, title: str, principles: List[str], source_experiences: List[str]) -> Dict:
        doctrine_id = f"DOCTRINE-{uuid.uuid4().hex[:8]}"
        doctrine = {
            "id": doctrine_id,
            "timestamp": datetime.utcnow().isoformat(),
            "title": title,
            "principles": principles,
            "source_experiences": source_experiences
        }
        self.doctrines.append(doctrine)
        return doctrine

    def list_doctrines(self) -> List[Dict]:
        return self.doctrines

    def find_doctrine_by_principle(self, principle: str) -> List[Dict]:
        return [d for d in self.doctrines if principle in d["principles"]]
