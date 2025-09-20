from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
import uuid
import json
import math
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class AffinityVector:
    """Multi-dimensional affinity representation"""
    semantic: float = 0.0      # shared meaning/concepts
    emotional: float = 0.0     # emotional resonance
    temporal: float = 0.0      # time-based clustering
    contextual: float = 0.0    # situational similarity
    causal: float = 0.0        # cause-effect relationships
    
    @property
    def composite(self) -> float:
        """Weighted composite affinity score"""
        return (0.3 * self.semantic + 0.25 * self.emotional + 
                0.2 * self.temporal + 0.15 * self.contextual + 
                0.1 * self.causal)

class MemoryAffinityMappingEngine:
    def __init__(self):
        self.affinity_log: List[Dict] = []
        self.memory_graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.tag_clusters: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_windows = {
            "immediate": timedelta(minutes=5),
            "recent": timedelta(hours=2), 
            "same_day": timedelta(days=1),
            "contextual": timedelta(days=7)
        }

    def map_affinity(self, memory_a_id: str, memory_b_id: str, 
                    memory_a: Dict, memory_b: Dict,
                    relationship_type: str = "semantic") -> Dict:
        """Enhanced affinity mapping with multi-dimensional analysis"""
        
        affinity_id = f"MEMMAP-{uuid.uuid4().hex[:8]}"
        vector = self._calculate_affinity_vector(memory_a, memory_b)
        
        # Determine dominant affinity dimension
        dimensions = {
            "semantic": vector.semantic,
            "emotional": vector.emotional, 
            "temporal": vector.temporal,
            "contextual": vector.contextual,
            "causal": vector.causal
        }
        dominant_dimension = max(dimensions.items(), key=lambda x: x[1])
        
        mapping = {
            "id": affinity_id,
            "timestamp": datetime.utcnow().isoformat(),
            "memory_a_id": memory_a_id,
            "memory_b_id": memory_b_id,
            "relationship_type": relationship_type,
            "affinity_vector": {
                "semantic": vector.semantic,
                "emotional": vector.emotional,
                "temporal": vector.temporal, 
                "contextual": vector.contextual,
                "causal": vector.causal,
                "composite": vector.composite
            },
            "dominant_dimension": dominant_dimension[0],
            "strength": self._categorize_strength(vector.composite),
            "bidirectional": self._is_bidirectional(relationship_type)
        }
        
        self.affinity_log.append(mapping)
        
        # Update memory graph
        self.memory_graph[memory_a_id][memory_b_id] = vector.composite
        if mapping["bidirectional"]:
            self.memory_graph[memory_b_id][memory_a_id] = vector.composite
            
        # Update tag clusters
        self._update_tag_clusters(memory_a, memory_b, vector.semantic)
        
        return mapping

    def _calculate_affinity_vector(self, mem_a: Dict, mem_b: Dict) -> AffinityVector:
        """Calculate multi-dimensional affinity between two memories"""
        
        # Semantic affinity (tag overlap + text similarity)
        semantic = self._semantic_affinity(mem_a, mem_b)
        
        # Emotional affinity
        emotional = self._emotional_affinity(mem_a, mem_b)
        
        # Temporal affinity 
        temporal = self._temporal_affinity(mem_a, mem_b)
        
        # Contextual affinity
        contextual = self._contextual_affinity(mem_a, mem_b)
        
        # Causal affinity (if one led to the other)
        causal = self._causal_affinity(mem_a, mem_b)
        
        return AffinityVector(semantic, emotional, temporal, contextual, causal)

    def _semantic_affinity(self, mem_a: Dict, mem_b: Dict) -> float:
        """Calculate semantic similarity based on tags and content"""
        tags_a = set(mem_a.get("tags", {}).keys())
        tags_b = set(mem_b.get("tags", {}).keys())
        
        if not tags_a and not tags_b:
            return 0.0
            
        # Jaccard similarity for tags
        intersection = len(tags_a.intersection(tags_b))
        union = len(tags_a.union(tags_b))
        tag_similarity = intersection / union if union > 0 else 0.0
        
        # Simple text similarity (could be enhanced with embeddings)
        text_a = mem_a.get("text", "").lower().split()
        text_b = mem_b.get("text", "").lower().split()
        text_intersection = len(set(text_a).intersection(set(text_b)))
        text_union = len(set(text_a).union(set(text_b)))
        text_similarity = text_intersection / text_union if text_union > 0 else 0.0
        
        return round(0.7 * tag_similarity + 0.3 * text_similarity, 3)

    def _emotional_affinity(self, mem_a: Dict, mem_b: Dict) -> float:
        """Calculate emotional similarity"""
        emotion_a = mem_a.get("emotion", "").lower()
        emotion_b = mem_b.get("emotion", "").lower()
        
        if not emotion_a or not emotion_b:
            return 0.0
            
        # Exact match
        if emotion_a == emotion_b:
            intensity_a = mem_a.get("intensity", 0.0)
            intensity_b = mem_b.get("intensity", 0.0)
            # Similarity decreases with intensity difference
            intensity_diff = abs(intensity_a - intensity_b)
            return round(1.0 - (intensity_diff * 0.5), 3)
        
        # Emotional family matching (could be expanded)
        positive_emotions = {"joy", "love", "hope", "wonder", "flow"}
        negative_emotions = {"grief", "anger", "fear", "sad", "anxiety"}
        
        if (emotion_a in positive_emotions and emotion_b in positive_emotions) or \
           (emotion_a in negative_emotions and emotion_b in negative_emotions):
            return 0.4
            
        return 0.0

    def _temporal_affinity(self, mem_a: Dict, mem_b: Dict) -> float:
        """Calculate temporal clustering strength"""
        try:
            ts_a = datetime.fromisoformat(mem_a.get("timestamp", "").replace('Z', '+00:00'))
            ts_b = datetime.fromisoformat(mem_b.get("timestamp", "").replace('Z', '+00:00'))
            
            time_diff = abs((ts_a - ts_b).total_seconds())
            
            # Exponential decay based on time difference
            for window_name, window_delta in self.temporal_windows.items():
                if time_diff <= window_delta.total_seconds():
                    if window_name == "immediate":
                        return 0.9
                    elif window_name == "recent":
                        return 0.7
                    elif window_name == "same_day":
                        return 0.5
                    else:  # contextual
                        return 0.3
            
            # Decay for longer periods
            days = time_diff / 86400
            return round(max(0.0, 0.3 * math.exp(-days / 30)), 3)
            
        except (ValueError, TypeError):
            return 0.0

    def _contextual_affinity(self, mem_a: Dict, mem_b: Dict) -> float:
        """Calculate contextual similarity (situation, environment)"""
        # Look for context tags (source/, context/, situation/)
        context_prefixes = ["source/", "context/", "situation/", "location/"]
        
        contexts_a = []
        contexts_b = []
        
        for tag in mem_a.get("tags", {}):
            for prefix in context_prefixes:
                if tag.startswith(prefix):
                    contexts_a.append(tag)
                    
        for tag in mem_b.get("tags", {}):
            for prefix in context_prefixes:
                if tag.startswith(prefix):
                    contexts_b.append(tag)
        
        if not contexts_a or not contexts_b:
            return 0.0
            
        # Jaccard similarity for context tags
        intersection = len(set(contexts_a).intersection(set(contexts_b)))
        union = len(set(contexts_a).union(set(contexts_b)))
        
        return round(intersection / union if union > 0 else 0.0, 3)

    def _causal_affinity(self, mem_a: Dict, mem_b: Dict) -> float:
        """Calculate causal relationship strength"""
        # Look for causal indicators in text
        causal_indicators = ["because", "led to", "caused", "resulted in", "triggered"]
        
        text_a = mem_a.get("text", "").lower()
        text_b = mem_b.get("text", "").lower()
        
        # Check for causal language
        causal_score = 0.0
        for indicator in causal_indicators:
            if indicator in text_a or indicator in text_b:
                causal_score += 0.2
                
        # Check for causal tags
        tags_a = mem_a.get("tags", {})
        tags_b = mem_b.get("tags", {})
        
        for tag in list(tags_a.keys()) + list(tags_b.keys()):
            if any(cause_word in tag for cause_word in ["cause", "effect", "trigger", "lead"]):
                causal_score += 0.3
                
        return round(min(1.0, causal_score), 3)

    def _categorize_strength(self, composite_score: float) -> str:
        """Categorize affinity strength"""
        if composite_score >= 0.8:
            return "very_strong"
        elif composite_score >= 0.6:
            return "strong" 
        elif composite_score >= 0.4:
            return "moderate"
        elif composite_score >= 0.2:
            return "weak"
        else:
            return "very_weak"

    def _is_bidirectional(self, relationship_type: str) -> bool:
        """Determine if relationship is bidirectional"""
        bidirectional_types = {"semantic", "emotional", "temporal", "contextual"}
        return relationship_type in bidirectional_types

    def _update_tag_clusters(self, mem_a: Dict, mem_b: Dict, semantic_score: float):
        """Update tag clustering based on co-occurrence"""
        if semantic_score < 0.3:
            return
            
        tags_a = mem_a.get("tags", {}).keys()
        tags_b = mem_b.get("tags", {}).keys()
        
        # Add co-occurring tags to clusters
        for tag_a in tags_a:
            for tag_b in tags_b:
                if tag_a != tag_b:
                    self.tag_clusters[tag_a].add(tag_b)
                    self.tag_clusters[tag_b].add(tag_a)

    def find_memory_neighbors(self, memory_id: str, min_affinity: float = 0.3, 
                             limit: int = 10) -> List[Tuple[str, float]]:
        """Find memories with strongest affinity to given memory"""
        if memory_id not in self.memory_graph:
            return []
            
        neighbors = [(mem_id, score) for mem_id, score in self.memory_graph[memory_id].items() 
                    if score >= min_affinity]
        return sorted(neighbors, key=lambda x: x[1], reverse=True)[:limit]

    def find_memory_clusters(self, min_cluster_size: int = 3, 
                           min_affinity: float = 0.4) -> List[List[str]]:
        """Find clusters of highly connected memories"""
        visited = set()
        clusters = []
        
        for memory_id in self.memory_graph:
            if memory_id in visited:
                continue
                
            cluster = self._dfs_cluster(memory_id, min_affinity, visited)
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
                
        return clusters

    def _dfs_cluster(self, start_id: str, min_affinity: float, visited: set) -> List[str]:
        """Depth-first search to find connected memory cluster"""
        if start_id in visited:
            return []
            
        visited.add(start_id)
        cluster = [start_id]
        
        for neighbor_id, affinity in self.memory_graph.get(start_id, {}).items():
            if affinity >= min_affinity and neighbor_id not in visited:
                cluster.extend(self._dfs_cluster(neighbor_id, min_affinity, visited))
                
        return cluster

    def get_affinity_insights(self) -> Dict:
        """Generate insights about memory affinity patterns"""
        if not self.affinity_log:
            return {"total_mappings": 0}
            
        # Dimension analysis
        dimensions = defaultdict(list)
        strengths = defaultdict(int)
        
        for mapping in self.affinity_log:
            vector = mapping["affinity_vector"]
            dominant = mapping["dominant_dimension"]
            strength = mapping["strength"]
            
            dimensions[dominant].append(vector["composite"])
            strengths[strength] += 1
            
        avg_by_dimension = {dim: sum(scores)/len(scores) 
                          for dim, scores in dimensions.items()}
        
        return {
            "total_mappings": len(self.affinity_log),
            "avg_affinity_by_dimension": avg_by_dimension,
            "strength_distribution": dict(strengths),
            "tag_clusters": {tag: len(cluster) for tag, cluster in self.tag_clusters.items()},
            "most_connected_memories": self._find_most_connected_memories()
        }

    def _find_most_connected_memories(self, top_k: int = 5) -> List[Tuple[str, int]]:
        """Find memories with most connections"""
        connection_counts = {mem_id: len(connections) 
                           for mem_id, connections in self.memory_graph.items()}
        return sorted(connection_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def export_graph(self, path: str, format: str = "json") -> None:
        """Export memory graph in various formats"""
        if format == "json":
            graph_data = {
                "nodes": list(self.memory_graph.keys()),
                "edges": [
                    {"source": src, "target": tgt, "weight": weight}
                    for src, targets in self.memory_graph.items()
                    for tgt, weight in targets.items()
                ],
                "metadata": self.get_affinity_insights()
            }
            with open(path, "w") as f:
                json.dump(graph_data, f, indent=2)
        elif format == "dot":
            # GraphViz DOT format for visualization
            self._export_dot(path)

    def _export_dot(self, path: str):
        """Export as GraphViz DOT file"""
        with open(path, "w") as f:
            f.write("digraph MemoryAffinityGraph {\n")
            f.write("  node [shape=ellipse];\n")
            
            for src, targets in self.memory_graph.items():
                for tgt, weight in targets.items():
                    if weight > 0.3:  # Only show significant connections
                        f.write(f'  "{src[:8]}" -> "{tgt[:8]}" [weight={weight:.2f}];\n')
            
            f.write("}\n")

    def list_affinities(self, filter_by: Optional[Dict] = None) -> List[Dict]:
        """List affinities with optional filtering"""
        if not filter_by:
            return self.affinity_log
            
        filtered = []
        for mapping in self.affinity_log:
            match = True
            for key, value in filter_by.items():
                if key == "min_affinity" and mapping["affinity_vector"]["composite"] < value:
                    match = False
                    break
                elif key == "dominant_dimension" and mapping["dominant_dimension"] != value:
                    match = False
                    break
                elif key == "strength" and mapping["strength"] != value:
                    match = False
                    break
            if match:
                filtered.append(mapping)
                
        return filtered

    def export_affinities(self, path: str, include_insights: bool = True) -> None:
        """Enhanced export with optional insights"""
        export_data = {
            "affinities": self.affinity_log,
            "graph": dict(self.memory_graph),
            "tag_clusters": {tag: list(cluster) for tag, cluster in self.tag_clusters.items()}
        }
        
        if include_insights:
            export_data["insights"] = self.get_affinity_insights()
            
        with open(path, "w") as f:
            json.dump(export_data, f, indent=2)