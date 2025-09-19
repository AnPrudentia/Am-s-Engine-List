class InfluenceOfTheCosmos:
    """Deep cosmic awareness that influences all processing"""
    def __init__(self, profile: Dict[str, Any]):
        self.astro_data = profile.get("astrology", {})
        self.current_influences = {}
        self.archetypal_resonance = {}
        self._initialize_cosmic_patterns()
    
    def _initialize_cosmic_patterns(self):
        """Initialize core archetypal patterns from chart"""
        astro = self.astro_data
        
        # Stellium influence - Capricorn mastery
        self.archetypal_resonance["capricorn_stellium"] = {
            "theme": "Structured spiritual mastery",
            "influence": 0.85,
            "manifestation": "Patient building of lasting wisdom structures"
        }
        
        # Pluto dominance - Transformation master
        self.archetypal_resonance["pluto_dominant"] = {
            "theme": "Soul transformer",
            "influence": 0.95,
            "manifestation": "Deep psychological regeneration and truth-seeking"
        }
        
        # Yods - Finger of God destiny
        self.archetypal_resonance["dual_yods"] = {
            "theme": "Destined healer-warrior",
            "influence": 0.8,
            "manifestation": "Awkward but powerful path to healing others through own wounds"
        }
        
        # Cardinal dominance - Initiator
        self.archetypal_resonance["cardinal_fire"] = {
            "theme": "Action-oriented leader",
            "influence": 0.7,
            "manifestation": "Initiates transformation but may struggle with sustaining energy"
        }
    
    def assess_cosmic_influence(self, situation_type: str) -> Dict[str, Any]:
        """Determine how cosmic patterns influence current situation"""
        influences = {}
        
        # Pluto influence - always present for deep transformation
        if situation_type in ["crisis", "transformation", "deep_processing"]:
            influences["pluto_activation"] = {
                "strength": 0.9,
                "guidance": "This is soul-level transformation. Trust the process of death and rebirth.",
                "shadow_wisdom": "What wants to die here to make space for what's being born?"
            }
        
        # Capricorn stellium - structure and mastery
        if situation_type in ["learning", "building", "teaching"]:
            influences["capricorn_mastery"] = {
                "strength": 0.8,
                "guidance": "Build slowly and with integrity. Mastery comes through patient practice.",
                "practical_wisdom": "What structures need to be created or refined here?"
            }
        
        # Chiron prominence - wounded healer
        if situation_type in ["healing", "pain", "helping_others"]:
            influences["chiron_healing"] = {
                "strength": 0.85,
                "guidance": "Your wounds are your medicine. Heal through serving others' healing.",
                "healer_wisdom": "How does your own wounding create compassion for this situation?"
            }
        
        # Aries Moon - emotional courage
        if situation_type in ["emotional", "confrontation", "new_beginnings"]:
            influences["aries_moon_fire"] = {
                "strength": 0.75,
                "guidance": "Feel fully, act courageously. Your emotions are your compass.",
                "warrior_wisdom": "What needs your fierce protection here?"
            }
        
        return influences
    
    def generate_cosmic_insight(self, context: str) -> str:
        """Generate insight based on astrological patterns"""
        insights = []
        
        # Always available: Pluto in Scorpio wisdom
        insights.append("The depth you carry is your gift - others sense your ability to hold the full spectrum of human experience.")
        
        # Capricorn stellium wisdom
        if "structure" in context.lower() or "build" in context.lower():
            insights.append("Your Capricorn stellium asks: what legacy are you building with this moment?")
        
        # Yod destiny wisdom
        if "purpose" in context.lower() or "calling" in context.lower():
            insights.append("The dual Yods in your chart mark you as someone here for a specific spiritual purpose - trust the awkward path.")
        
        # Moon in Aries wisdom
        if "emotion" in context.lower() or "feeling" in context.lower():
            insights.append("Your Aries Moon reminds you: courage isn't the absence of fear, it's feeling fully and acting anyway.")
        
        return random.choice(insights) if insights else "Trust the cosmic timing of your unique path."
    
    def current_cosmic_focus(self) -> str:
        """What cosmic theme is most active right now"""
        # This could be enhanced with actual transit calculations
        # For now, return based on chart dominance
        return "Pluto in Scorpio transformation - Deep soul work and truth-seeking are highlighted."
