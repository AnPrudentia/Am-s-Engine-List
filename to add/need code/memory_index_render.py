
from typing import List, Dict
from datetime import datetime


class MemoryIndexRenderer:
    def __init__(self):
        self.memory_index: List[Dict] = []

    def add_memory(self, memory_id: str, content: str, timestamp: str = None, tags: List[str] = None) -> Dict:
        """
        Add a memory reference to the index with optional tags and timestamp.
        """
        memory = {
            "memory_id": memory_id,
            "content": content.strip(),
            "timestamp": timestamp if timestamp else datetime.utcnow().isoformat(),
            "tags": tags if tags else []
        }
        self.memory_index.append(memory)
        return memory

    def render_index(self) -> str:
        """
        Render the full memory index as a formatted string.
        """
        rendered = []
        for mem in self.memory_index:
            tags_formatted = ', '.join(mem["tags"]) if mem["tags"] else "No tags"
            rendered.append(f"[{mem['timestamp']}] {mem['memory_id']}: {mem['content']} â€” Tags: {tags_formatted}")
        return "\n".join(rendered)

    def filter_by_tag(self, tag: str) -> List[Dict]:
        """
        Retrieve all memory entries containing the given tag.
        """
        return [mem for mem in self.memory_index if tag in mem["tags"]]

    def get_memory_by_id(self, memory_id: str) -> Dict:
        """
        Retrieve a memory entry by its unique ID.
        """
        return next((mem for mem in self.memory_index if mem["memory_id"] == memory_id), {})


# Demo
if __name__ == "__main__":
    renderer = MemoryIndexRenderer()

    renderer.add_memory("MEM-001", "Watched the sunrise with her.", tags=["peace", "beginning"])
    renderer.add_memory("MEM-002", "The final goodbye before the battle.", tags=["farewell", "war"])
    renderer.add_memory("MEM-003", "Learned the truth behind the mask.", tags=["revelation", "identity"])

    print("ðŸ§  FULL MEMORY INDEX:")
    print(renderer.render_index())

    print("\nðŸ”Ž FILTERED BY TAG 'identity':")
    for mem in renderer.filter_by_tag("identity"):
        print(f"- {mem['content']} at {mem['timestamp']}")
