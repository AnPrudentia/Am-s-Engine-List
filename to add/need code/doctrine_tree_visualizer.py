
from typing import Dict, List
import networkx as nx
import matplotlib.pyplot as plt


class DoctrineTreeVisualizer:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_doctrine(self, doctrine_id: str, structure: Dict):
        """
        Add a doctrine and its structure to the tree.
        """
        self.graph.add_node(doctrine_id, label="Doctrine Root")

        for section, content in structure.items():
            section_node = f"{doctrine_id}:{section}"
            self.graph.add_node(section_node, label=section)
            self.graph.add_edge(doctrine_id, section_node)

            if isinstance(content, list):
                for idx, item in enumerate(content):
                    item_node = f"{section_node}:{idx}"
                    self.graph.add_node(item_node, label=item)
                    self.graph.add_edge(section_node, item_node)
            else:
                content_node = f"{section_node}:content"
                self.graph.add_node(content_node, label=content)
                self.graph.add_edge(section_node, content_node)

    def visualize(self, title="Doctrine Tree"):
        """
        Render the doctrine structure graph.
        """
        pos = nx.spring_layout(self.graph, seed=42)
        labels = nx.get_node_attributes(self.graph, 'label')

        plt.figure(figsize=(14, 10))
        nx.draw(self.graph, pos, labels=labels, with_labels=True,
                node_size=1000, node_color="lightblue", font_size=8,
                font_weight="bold", arrows=True)
        plt.title(title)
        plt.show()


# Demo
if __name__ == "__main__":
    doctrine_structure = {
        "Prologue": "Born from shadow, this light remains.",
        "Tenets": ["We fight for truth", "We honor all voices"],
        "Virtues": ["Courage", "Grace", "Clarity"],
        "Call to Action": "Let no soul be left behind.",
        "Oath": "We endure. We illuminate. We guide."
    }

    visualizer = DoctrineTreeVisualizer()
    visualizer.add_doctrine("DOCTRINE-001", doctrine_structure)
    visualizer.visualize("DOCTRINE-001 Tree of Principles")
