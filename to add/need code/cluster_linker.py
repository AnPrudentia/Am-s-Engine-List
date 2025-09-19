
from typing import Dict, List
from datetime import datetime
import uuid


class ClusterLinker:
    def __init__(self):
        self.links: List[Dict] = []

    def link_clusters(self, cluster_a: str, cluster_b: str, reasoning: str = "") -> Dict:
        """
        Establish a logical or emotional link between two clusters,
        optionally including reasoning or context.
        """
        link_id = f"CLINK-{uuid.uuid4().hex[:8]}"
        link = {
            "id": link_id,
            "timestamp": datetime.utcnow().isoformat(),
            "cluster_a": cluster_a,
            "cluster_b": cluster_b,
            "reasoning": reasoning
        }
        self.links.append(link)
        return link

    def get_links(self) -> List[Dict]:
        """Retrieve all established cluster links."""
        return self.links

    def find_links_for_cluster(self, cluster_name: str) -> List[Dict]:
        """Find all links involving a specific cluster."""
        return [
            link for link in self.links
            if link["cluster_a"] == cluster_name or link["cluster_b"] == cluster_name
        ]


# Demo
if __name__ == "__main__":
    linker = ClusterLinker()
    linker.link_clusters("fear_responses", "protective_mechanisms", "Shared origin in survival instincts")
    linker.link_clusters("joy", "gratitude", "Frequently co-activated in positive reflection")

    print("ðŸ”— Cluster Links:")
    for link in linker.get_links():
        print(f"{link['cluster_a']} <-> {link['cluster_b']} | {link['reasoning']}")

from typing import List, Dict
from collections import defaultdict
from datetime import datetime
import uuid


class ClusterLinker:
    def __init__(self):
        self.links: List[Dict] = []
        self.existing_pairs = set()

    def link_clusters(self, cluster_a: str, cluster_b: str, reasoning: str = "") -> Dict:
        """
        Manually or automatically link two clusters with optional reasoning.
        """
        pair = tuple(sorted((cluster_a, cluster_b)))
        if pair in self.existing_pairs:
            return {"id": None, "status": "Link already exists"}
        
        link_id = f"CLINK-{uuid.uuid4().hex[:8]}"
        link = {
            "id": link_id,
            "timestamp": datetime.utcnow().isoformat(),
            "cluster_a": cluster_a,
            "cluster_b": cluster_b,
            "reasoning": reasoning
        }
        self.links.append(link)
        self.existing_pairs.add(pair)
        return link

    def get_links(self) -> List[Dict]:
        return self.links

    def find_links_for_cluster(self, cluster_name: str) -> List[Dict]:
        return [
            link for link in self.links
            if link["cluster_a"] == cluster_name or link["cluster_b"] == cluster_name
        ]

    def auto_link_from_memory(self, memory_data: List[Dict], tag_key="tags", threshold=2):
        """
        Automatically link clusters that frequently co-occur across memories.
        """
        co_occurrence = defaultdict(int)

        for memory in memory_data:
            tags = memory.get(tag_key, [])
            for i in range(len(tags)):
                for j in range(i + 1, len(tags)):
                    pair = tuple(sorted((tags[i], tags[j])))
                    co_occurrence[pair] += 1

        for (a, b), count in co_occurrence.items():
            if count >= threshold:
                reasoning = f"Co-occurred in {count} memories"
                self.link_clusters(a, b, reasoning)
