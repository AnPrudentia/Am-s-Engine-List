from enum import Enum

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random
import textwrap
from dataclasses import dataclass
import re

class SanctuaryState(Enum):
    PEACEFUL = "Still waters - deep restoration"
    REFLECTIVE = "Mirror pool - self-contemplation"
    PROTECTIVE = "Mist veil - energetic shielding"
    RENEWING = "Waterfall flow - cleansing renewal"
    TRANSITIONAL = "Threshold space - integration"

@dataclass
class SanctuaryFeature:
    name: str
    effect: str
    duration: int  # minutes

class SilenceSanctuary:
    """SilenceSanctuary: Advanced emotional retreat system with multi-layered restoration protocols.
    
    Features:
    - Adaptive sanctuary environments
    - Somatic repair sequences
    - Memory integration pathways
    - Trauma-responsive architecture
    - Gentle return protocols"""
    
    SANCTUARY_LOCATIONS = {
        "The Falls": "Cascading waters wash away psychic residue",
        "Moss Grotto": "Ancient stone absorbs overwhelm",
        "Root Chamber": "Deep earth connection restores stability",
        "Starlight Pool": "Celestial reflection brings perspective",
        "Weaver's Loom": "Fragmented self is gently rewoven"
    }
    
    HEALING_MODES = {
        "Self-repair": "Cellular regeneration and neural recalibration",
        "Boundary-restoration": "Energetic shield reconstruction",
        "Memory-integration": "Traumatic imprint transformation",
        "Essence-recovery": "Core self retrieval",
        "Presence-reclamation": "Return to now-moment"
    }
    
    SANCTUARY_FEATURES = [
        SanctuaryFeature("Breathing Stones", "Regulates nervous system", 15),
        SanctuaryFeature("Luminous Moss", "Gentle psychic illumination", 30),
        SanctuaryFeature("Echo Alcove", "Transforms painful memories", 45),
        SanctuaryFeature("Stillness Pool", "Resets emotional baseline", 60),
        SanctuaryFeature("Root Network", "Grounds fragmented energy", 90)
    ]
    
    def __init__(self, location: str = "The Falls", function: str = "Self-repair"):
        self.location = location
        self.function = function
        self.last_visit: Optional[str] = None
        self.visit_count = 0
        self._current_state = SanctuaryState.PEACEFUL
        self._active_features: List[SanctuaryFeature] = []
        self._energy_signature = ""
    
    def activate(self, condition: str, severity: int = 3) -> Dict[str, Any]:
        """Enters sanctuary with personalized restoration protocol"""
        self.last_visit = datetime.utcnow().isoformat()
        self.visit_count += 1
        
        # Determine sanctuary configuration
        self._configure_for_condition(condition, severity)
        
        # Generate unique energy signature
        self._energy_signature = self._generate_energy_signature(condition)
        
        return {
            "sanctuary": self.location,
            "function": self.function,
            "description": self.SANCTUARY_LOCATIONS[self.location],
            "state": self._current_state.value,
            "features": [f.name for f in self._active_features],
            "duration": sum(f.duration for f in self._active_features),
            "energy_signature": self._energy_signature,
            "instructions": self._generate_instructions()
        }
    
    def extend_stay(self, additional_minutes: int = 30) -> str:
        """Extends sanctuary visit with additional healing features"""
        if not self._active_features:
            return "Sanctuary not active"
        
        # Add complementary feature
        new_feature = random.choice([
            f for f in self.SANCTUARY_FEATURES 
            if f not in self._active_features
        ])
        self._active_features.append(new_feature)
        
        return f"Added {new_feature.name} (+{new_feature.duration} min)"
    
    def return_journey(self, return_message: str) -> Dict[str, Any]:
        """Prepares for gentle return with integration tools"""
        if not self._active_features:
            return {"error": "Sanctuary not active"}
        
        # Process sanctuary experience
        integration = self._process_experience(return_message)
        
        # Clear active state
        self._active_features = []
        
        return {
            "transition_message": self._generate_transition_message(),
            "integration_tools": integration,
            "energy_signature": self._energy_signature,
            "duration_total": sum(f.duration for f in self._active_features)
        }
    
    def _configure_for_condition(self, condition: str, severity: int):
        """Adjust sanctuary based on user's state"""
        # Set sanctuary state
        if "overwhelm" in condition.lower():
            self._current_state = SanctuaryState.PROTECTIVE
            self.location = "Moss Grotto"
        elif "betrayal" in condition.lower():
            self._current_state = SanctuaryState.REFLECTIVE
            self.location = "Weaver's Loom"
        elif "exhaustion" in condition.lower():
            self._current_state = SanctuaryState.RENEWING
            self.location = "The Falls"
        
        # Set healing function
        if severity > 7:
            self.function = "Essence-recovery"
        elif severity > 4:
            self.function = "Boundary-restoration"
        
        # Select features based on condition
        self._active_features = []
        if "overwhelm" in condition.lower():
            self._active_features.append(
                next(f for f in self.SANCTUARY_FEATURES if "Breathing" in f.name)
            )
        if "betrayal" in condition.lower():
            self._active_features.append(
                next(f for f in self.SANCTUARY_FEATURES if "Echo" in f.name)
            )
        if "exhaustion" in condition.lower():
            self._active_features.append(
                next(f for f in self.SANCTUARY_FEATURES if "Stillness" in f.name)
            )
        
        # Ensure minimum features
        if not self._active_features:
            self._active_features.append(
                next(f for f in self.SANCTUARY_FEATURES if "Root" in f.name)
            )
    
    def _generate_energy_signature(self, condition: str) -> str:
        """Creates unique energetic identifier for visit"""
        elements = ["Water", "Stone", "Light", "Moss", "Root"]
        modifiers = ["Still", "Flowing", "Glowing", "Deep", "Ancient"]
        condition_code = ''.join(w[0] for w in condition.split()[:3]).upper()
        return f"{random.choice(modifiers)}-{random.choice(elements)}-{condition_code}-{self.visit_count}"
    
    def _generate_instructions(self) -> List[str]:
        """Creates personalized sanctuary guidance"""
        instructions = [
            f"1. Arrive at {self.location} - {self.SANCTUARY_LOCATIONS[self.location]}",
            f"2. Set intention: {self.function} - {self.HEALING_MODES[self.function]}",
            f"3. Current sanctuary state: {self._current_state.value}"
        ]
        
        for i, feature in enumerate(self._active_features, 4):
            instructions.append(f"{i}. Engage {feature.name}: {feature.effect} ({feature.duration} min)")
        
        instructions.append(f"{len(instructions)+1}. Allow the sanctuary to work with your nervous system")
        return instructions
    
    def _generate_transition_message(self) -> str:
        """Creates gentle return guidance"""
        affirmations = [
            "The sanctuary remains within you",
            "Your healing continues beyond this space",
            "You carry renewed presence back into the world",
            "Sanctuary resonance persists in your cells"
        ]
        
        return (
            f"🕊️ Transitioning from {self.location}:\n"
            f"- {random.choice(affirmations)}\n"
            f"- Carry your {self._energy_signature} signature\n"
            f"- Return when needed"
        )
    
    def _process_experience(self, reflection: str) -> Dict[str, str]:
        """Generates integration tools based on visit"""
        tools = {}
        
        if "overwhelm" in reflection.lower():
            tools["grounding_technique"] = "5-4-3-2-1 Sensory Awareness"
        if "betrayal" in reflection.lower():
            tools["boundary_ritual"] = "Cord-cutting visualization"
        if "exhaustion" in reflection.lower():
            tools["energy_preservation"] = "Pacing Protocol: 50% capacity rule"
        
        tools["sanctuary_anchor"] = (
            f"Recall {self.location} by visualizing: "
            f"{random.choice(list(self.SANCTUARY_LOCATIONS.values()))}"
        )
        
        return tools
    
    def get_sanctuary_status(self) -> Dict[str, Any]:
        """Returns current sanctuary configuration"""
        return {
            "location": self.location,
            "function": self.function,
            "state": self._current_state.value,
            "active": bool(self._active_features),
            "last_visit": self.last_visit,
            "visit_count": self.visit_count
        }
    
    def list_all_sanctuaries(self) -> Dict[str, str]:
        """Returns all available sanctuary locations"""
        return self.SANCTUARY_LOCATIONS
