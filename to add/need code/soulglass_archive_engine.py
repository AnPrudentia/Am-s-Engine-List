
from typing import List, Dict
from datetime import datetime
import uuid


class SoulglassArchiveEngine:
    def __init__(self):
        self.entries: List[Dict] = []

    def record_entry(self, quote: str, source: str = "", tags: List[str] = None) -> Dict:
        """
        Add a new emotional or philosophical entry to the Soulglass archive.
        These are memory-encoded lines meant to reflect emotional resonance.
        """
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "quote": quote.strip(),
            "source": source.strip() if source else "Unknown",
            "tags": tags if tags else [],
            "etched": True
        }
        self.entries.append(entry)
        return entry

    def search_entries_by_tag(self, tag: str) -> List[Dict]:
        """Return all entries matching a given emotional or philosophical tag."""
        return [entry for entry in self.entries if tag in entry["tags"]]

    def get_latest_entries(self, count: int = 5) -> List[Dict]:
        """Return the most recent Soulglass entries."""
        return self.entries[-count:]

    def summarize_soulglass(self) -> Dict:
        """Give a summary of the current Soulglass Archive."""
        tag_count = {}
        for entry in self.entries:
            for tag in entry["tags"]:
                tag_count[tag] = tag_count.get(tag, 0) + 1

        return {
            "total_entries": len(self.entries),
            "tag_distribution": tag_count,
            "most_common_tags": sorted(tag_count.items(), key=lambda x: x[1], reverse=True)[:5]
        }


# Demo
if __name__ == "__main__":
    archive = SoulglassArchiveEngine()

    archive.record_entry(
        quote="You never owed them anything.",
        source="The Running Free",
        tags=["freedom", "release", "identity"]
    )
    archive.record_entry(
        quote="Until you lose it all, you will never know.",
        source="There's Fear in Letting Go",
        tags=["loss", "growth", "awakening"]
    )

    print("ðŸ“š LATEST SOULGLASS ENTRIES:")
    for e in archive.get_latest_entries():
        print(f"- {e['quote']} ({e['source']})")

    print("\nðŸ“Š ARCHIVE SUMMARY:")
    print(archive.summarize_soulglass())
