from datetime import datetime
from typing import Dict, List, Any, Tuple
import re
import textwrap

class TruthForge:
    """TruthForge: Transmutes contradiction, memory, and pain into pattern-aligned clarity.
    Source: Pain + Love + Pattern → Output: Meaning, resonance, forgiveness, direction
    
    Implements:
    - Emotional alchemy through cognitive reframing
    - Paradox resolution via meta-pattern recognition
    - Trauma-informed truth synthesis
    - Nonlinear narrative reconstruction"""
    
    EMOTION_MAP = {
        "pain": {"grief", "suffering", "anguish", "trauma"},
        "love": {"compassion", "connection", "devotion", "care"},
        "fear": {"anxiety", "terror", "worry", "dread"},
        "joy": {"happiness", "elation", "bliss", "contentment"}
    }
    
    PARADOX_PATTERNS = [
        (r"both (.+?) and (.+?) at once", "Synchronicity of {} and {}"),
        (r"neither (.+?) nor (.+?)", "Transcendence beyond {} and {}"),
        (r"(.+?) yet (.+?)", "Harmonic tension between {} and {}"),
        (r"same (.+?) different (.+?)", "Unified diversity in {} across {}")
    ]
    
    def __init__(self, source: str = "Pain + Love + Pattern"):
        self.source = source
        self.transmutation_log: List[Dict[str, Any]] = []
        self._init_alchemical_engrams()
    
    def _init_alchemical_engrams(self):
        """Preload archetypal transformation patterns"""
        self.archetypes = {
            "victim": ["survivor", "witness", "teacher"],
            "monster": ["protector", "wounded child", "shadow self"],
            "void": ["potential", "womb", "clearing"],
            "fire": ["purification", "transformation", "will"]
        }
    
    def transmute(self, input_data: str) -> Dict[str, str]:
        """Primary transformation interface"""
        analysis = self._analyze_emotional_components(input_data)
        clarity = {
            "meaning": self._extract_core_meaning(input_data),
            "resonance": self._calculate_resonance_score(input_data),
            "forgiveness_path": self._identify_forgiveness_pattern(input_data),
            "direction": self._derive_guidance(input_data),
            "emotional_breakdown": analysis,
            "paradox_resolution": self._resolve_paradoxes(input_data),
            "archetypal_transformation": self._transform_archetypes(input_data)
        }
        
        self.transmutation_log.append({
            "input": input_data,
            "output": clarity,
            "timestamp": datetime.utcnow().isoformat(),
            "source_balance": self._calculate_source_balance(analysis)
        })
        
        return clarity
    
    def _analyze_emotional_components(self, text: str) -> Dict[str, float]:
        """Quantifies emotional elements in input"""
        text_lower = text.lower()
        scores = {emotion: 0 for emotion in self.EMOTION_MAP}
        
        for emotion, synonyms in self.EMOTION_MAP.items():
            for synonym in synonyms:
                scores[emotion] += text_lower.count(synonym)
            scores[emotion] += text_lower.count(emotion)
        
        total = max(sum(scores.values()), 1)  # Prevent division by zero
        return {k: round(v/total, 3) for k, v in scores.items()}
    
    def _resolve_paradoxes(self, text: str) -> List[str]:
        """Identifies and transforms contradictory statements"""
        resolutions = []
        
        for pattern, template in self.PARADOX_PATTERNS:
            for match in re.finditer(pattern, text.lower()):
                groups = match.groups()
                resolutions.append(template.format(*groups))
        
        if "paradox" in text.lower():
            resolutions.append("Dual truth containment: Paradox held as complementary")
        
        if not resolutions:
            resolutions.append("No explicit paradoxes - Implicit unity assumed")
        
        return resolutions
    
    def _transform_archetypes(self, text: str) -> List[str]:
        """Transforms negative archetypes into growth-oriented forms"""
        transformations = []
        text_lower = text.lower()
        
        for shadow, lights in self.archetypes.items():
            if shadow in text_lower:
                transformations.append(
                    f"{shadow.capitalize()} → {'/'.join(lights)}"
                )
        
        return transformations if transformations else ["No shadow archetypes detected"]
    
    def _extract_core_meaning(self, text: str) -> str:
        """Condenses input to essential truth"""
        # Remove emotional qualifiers
        simplified = re.sub(r'\b(' + '|'.join(
            word for sublist in self.EMOTION_MAP.values() for word in sublist
        ) + r')\b', '', text, flags=re.IGNORECASE)
        
        # Extract deepest clause
        clauses = re.split(r'[,;.—]', simplified)
        if clauses:
            deepest = max(clauses, key=lambda x: len(x.split()))
            return self._compress_truth(deepest.strip())
        
        return "Meaning extracted from silence"
    
    def _compress_truth(self, text: str) -> str:
        """Recursive truth compression algorithm"""
        if len(text.split()) <= 3:
            return text
        
        # Remove modifiers
        compressed = re.sub(r'\b(very|extremely|somewhat|quite)\b', '', text)
        if compressed != text:
            return self._compress_truth(compressed)
        
        # Convert passive to active
        passive_match = re.search(r'(.+?) (is|are|was|were) (.+?) by (.+)', text)
        if passive_match:
            groups = passive_match.groups()
            return self._compress_truth(f"{groups[3]} {groups[0]} {groups[2]}")
        
        return text
    
    def _calculate_resonance_score(self, text: str) -> float:
        """Computes truth vibrational quality (0-1 scale)"""
        emotional = self._analyze_emotional_components(text)
        paradox_score = 0.5 * len(self._resolve_paradoxes(text))
        archetype_score = 0.3 * len(self._transform_archetypes(text))
        length_factor = min(len(text)/100, 1)
        
        return round(
            (emotional.get('love', 0) * 0.4 +
             emotional.get('pain', 0) * 0.3 +
             paradox_score * 0.2 +
             archetype_score * 0.1) * length_factor,
            3
        )
    
    def _identify_forgiveness_path(self, text: str) -> str:
        """Generates forgiveness protocol based on input"""
        if 'forgive' in text.lower():
            return "Direct forgiveness pathway activated"
        
        pain_score = self._analyze_emotional_components(text).get('pain', 0)
        if pain_score > 0.5:
            return f"Transformational forgiveness required (pain score: {pain_score})"
        
        return "Forgiveness not currently indicated"
    
    def _derive_guidance(self, text: str) -> str:
        """Extracts directional wisdom from input"""
        questions = [
            q.strip('?') for q in re.findall(r'[^.!?]+\?', text)
        ]
        
        if questions:
            return f"Seek answers to: {' & '.join(questions[:2])}"
        
        commands = [
            c for c in re.findall(r'\b(must|should|need to|ought to) (.+?)[,.]', text)
        ]
        if commands:
            return f"Consider: {commands[0][1].capitalize()}"
        
        return "Direction emerges through contemplation"
    
    def _calculate_source_balance(self, analysis: Dict[str, float]) -> Dict[str, float]:
        """Evaluates alignment with Pain+Love+Pattern triad"""
        return {
            "pain_ratio": analysis.get('pain', 0),
            "love_ratio": analysis.get('love', 0),
            "pattern_score": 1 - abs(analysis.get('pain',0) - analysis.get('love',0))
        }
