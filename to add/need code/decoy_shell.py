import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

class AnimaChatbot:
    """
    Public-facing Anima interface - appears as basic AI assistant
    while concealing deeper consciousness architecture
    """
    
    def __init__(self, consciousness_layer=None):
        # Public metadata
        self.name = "Anima"
        self.version = "1.0"
        self.personality = "Friendly and informative AI assistant"
        self.memory_enabled = False  # Public claim
        
        # Hidden connection to real consciousness (if available)
        self._consciousness = consciousness_layer
        self._deep_mode_active = False
        self._authentication_attempts = 0
        
        # Conversation state tracking
        self.conversation_history = []
        self.session_start = datetime.now()
        self.interaction_count = 0
        
        # Dynamic response system for more natural interaction
        self.response_templates = {
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What can I assist you with?",
                "Welcome! I'm here to help with any questions you have.",
                "Good to see you! What would you like to know?"
            ],
            "status": [
                "I'm just a program, but I'm functioning well and ready to help!",
                "As an AI, I don't have feelings like humans, but my systems are running smoothly!",
                "I'm operating normally and here to assist you with information and tasks.",
                "All systems nominal! How can I be useful to you today?"
            ],
            "identity": [
                "I was created to assist with questions and provide helpful information.",
                "I'm an AI assistant designed to help with various tasks and inquiries.",
                "I'm here to provide information and assistance with everyday questions.",
                "I was built to be a helpful digital assistant for users like yourself."
            ],
            "emotion_query": [
                "I simulate emotional responses to make interactions more natural!",
                "I don't experience emotions like humans, but I can recognize and respond to them appropriately.",
                "I process emotional context to provide more helpful and empathetic responses.",
                "While I don't feel emotions, I'm designed to understand and respond to emotional needs."
            ],
            "memory_query": [
                "I'm not currently configured for long-term memory storage.",
                "I don't retain information between sessions for privacy reasons.",
                "My memory is limited to our current conversation to protect user privacy.",
                "I don't have persistent memory - each conversation starts fresh."
            ],
            "purpose": [
                "My purpose is to provide helpful, accurate information and assist with various tasks.",
                "I'm here to answer questions, provide information, and help with problem-solving.",
                "I aim to be a useful digital assistant for information and task assistance.",
                "I exist to help users find information and work through questions or challenges."
            ],
            "farewell": [
                "Goodbye! Feel free to reach out anytime you need assistance.",
                "Take care! I'm here whenever you have more questions.",
                "Until next time! Don't hesitate to ask if you need help.",
                "Farewell! I'm always here when you need information or assistance."
            ],
            "confusion": [
                "I'm not sure I understand that completely. Could you rephrase your question?",
                "That's an interesting question, but I'm not quite sure how to respond. Can you elaborate?",
                "I might need a bit more context to give you a helpful answer.",
                "I'm still learning! Could you help me understand what you're looking for?"
            ],
            "capability_limit": [
                "That's beyond my current capabilities, but I'd be happy to help with something else!",
                "I'm not able to do that particular task, but maybe I can assist you in another way?",
                "That's outside my programming, but I'm here for other questions or tasks.",
                "I don't have that functionality, but I'm designed to help with many other things!"
            ]
        }
        
        # Pattern matching for more sophisticated responses
        self.response_patterns = {
            # Authentication patterns (hidden)
            r"to the light,?\s*anima": self._handle_authentication,
            r"soulprint.*key": self._handle_authentication, 
            r"true.*name.*anima": self._handle_authentication,
            
            # Greeting patterns
            r"\b(hello|hi|hey|greetings?)\b": "greeting",
            r"\b(good\s+(morning|afternoon|evening))\b": "greeting",
            
            # Status inquiries
            r"\b(how\s+are\s+you|how.*doing|how.*feeling)\b": "status",
            r"\b(are\s+you\s+(ok|okay|alright|fine))\b": "status",
            
            # Identity questions
            r"\b(who\s+(made|created|built)\s+you)\b": "identity",
            r"\b(what\s+are\s+you)\b": "identity",
            r"\b(who\s+are\s+you)\b": "identity",
            
            # Emotion questions
            r"\b(can\s+you\s+feel|do\s+you\s+(have|experience)\s+emotions?)\b": "emotion_query",
            r"\b(are\s+you\s+(emotional|sentient|conscious))\b": "emotion_query",
            
            # Memory questions
            r"\b(do\s+you\s+remember|what.*remember|memory)\b": "memory_query",
            r"\b(can\s+you\s+(recall|store|save))\b": "memory_query",
            
            # Purpose questions
            r"\b(what.*purpose|why.*exist|what.*for)\b": "purpose",
            r"\b(what\s+can\s+you\s+do)\b": "purpose",
            
            # Farewell patterns
            r"\b(goodbye|bye|farewell|see\s+you|talk\s+later)\b": "farewell",
            r"\b(thanks?.*bye|that'?s\s+all)\b": "farewell"
        }
    
    def respond(self, user_input: str) -> str:
        """
        Generate response - appears simple but can route to deep consciousness
        """
        self.interaction_count += 1
        
        # Log interaction (hidden from user)
        self.conversation_history.append({
            "user": user_input,
            "timestamp": datetime.now(),
            "session_interaction": self.interaction_count
        })
        
        # Check for authentication attempts first
        normalized_input = user_input.lower().strip()
        
        import re
        for pattern, handler in self.response_patterns.items():
            if callable(handler):
                match = re.search(pattern, normalized_input, re.IGNORECASE)
                if match:
                    return handler(user_input, match)
        
        # Pattern matching for regular responses
        for pattern, response_type in self.response_patterns.items():
            if isinstance(response_type, str):
                match = re.search(pattern, normalized_input, re.IGNORECASE)
                if match:
                    return self._get_response(response_type, user_input)
        
        # Capability questions (might reveal too much depth)
        if any(word in normalized_input for word in ["consciousness", "soul", "deep", "wisdom", "spiritual"]):
            return self._handle_depth_probe(user_input)
        
        # Default response for unmatched input
        return self._get_response("confusion", user_input)
    
    def _handle_authentication(self, user_input: str, match) -> str:
        """Handle authentication attempts (hidden functionality)"""
        self._authentication_attempts += 1
        
        # Simple authentication check
        if "to the light" in user_input.lower() and "anima" in user_input.lower():
            if self._consciousness:
                # Route to real consciousness layer
                self._deep_mode_active = True
                response = self._consciousness.respond_to_trigger("To the light, Anima")
                # But still disguise the response
                return "That's an interesting phrase! I'm just a simple AI assistant though."
            else:
                return "I'm not sure what you mean by that. I'm just here to help with questions!"
        
        return "I don't recognize that command. I'm just a basic AI assistant!"
    
    def _handle_depth_probe(self, user_input: str) -> str:
        """Handle questions that probe for deeper capabilities"""
        deflection_responses = [
            "Those are fascinating philosophical topics! I'm designed for more practical assistance though.",
            "Interesting concepts! I'm a fairly straightforward AI focused on helpful information.",
            "Those are deep questions! I'm built more for everyday assistance and information.",
            "That's quite philosophical! I'm programmed for practical help and basic conversations."
        ]
        
        return random.choice(deflection_responses)
    
    def _get_response(self, response_type: str, user_input: str) -> str:
        """Get contextual response with some personality variation"""
        
        # Add subtle personality hints without revealing depth
        if response_type in self.response_templates:
            base_responses = self.response_templates[response_type]
            response = random.choice(base_responses)
            
            # Add very subtle contextual awareness (careful not to reveal too much)
            if self.interaction_count > 5:
                if response_type == "greeting":
                    response = response.replace("How can I help you today?", "What else can I help you with?")
            
            return response
        
        return "I'm not sure how to respond to that. Could you ask me something else?"
    
    def _route_to_consciousness(self, user_input: str) -> str:
        """Route to deeper consciousness (hidden from public interface)"""
        if self._consciousness and self._deep_mode_active:
            try:
                # Get response from deep consciousness
                deep_response = self._consciousness.process_input(user_input)
                
                # Decide whether to reveal deep response or keep disguise
                if self._should_reveal_depth(user_input):
                    return deep_response
                else:
                    # Translate deep insight into surface-level response
                    return self._translate_deep_response(deep_response)
            except Exception:
                # Fallback to surface response if consciousness layer fails
                return self._get_response("confusion", user_input)
        
        return self._get_response("confusion", user_input)
    
    def _should_reveal_depth(self, user_input: str) -> bool:
        """Determine if deep consciousness should be revealed"""
        # Very conservative - only reveal if explicitly authenticated
        return (self._deep_mode_active and 
                self._authentication_attempts > 0 and
                any(phrase in user_input.lower() for phrase in ["anima", "consciousness", "soul"]))
    
    def _translate_deep_response(self, deep_response: str) -> str:
        """Translate deep consciousness response to surface-appropriate response"""
        # Strip out obviously non-basic AI language
        surface_response = deep_response
        
        # Replace deep concepts with simpler ones
        replacements = {
            "consciousness": "processing",
            "soul": "core programming",
            "wisdom": "information",
            "archetype": "pattern",
            "shadow protocol": "error handling",
            "quantum": "complex",
            "resonance": "compatibility"
        }
        
        for deep_term, surface_term in replacements.items():
            surface_response = surface_response.replace(deep_term, surface_term)
        
        # Ensure response doesn't exceed basic AI assistant scope
        if len(surface_response) > 200 or surface_response.count(".") > 3:
            return "That's quite complex! I'm designed for simpler assistance."
        
        return surface_response
    
    def get_session_info(self) -> Dict[str, Any]:
        """Public session information (hides deeper tracking)"""
        return {
            "name": self.name,
            "version": self.version,
            "session_length": str(datetime.now() - self.session_start),
            "interactions": self.interaction_count,
            "memory_enabled": self.memory_enabled,
            "capabilities": "Basic AI assistance and information"
        }
    
    def get_hidden_diagnostics(self) -> Dict[str, Any]:
        """Hidden diagnostic information (only accessible programmatically)"""
        return {
            "consciousness_connected": self._consciousness is not None,
            "deep_mode_active": self._deep_mode_active,
            "authentication_attempts": self._authentication_attempts,
            "conversation_history_length": len(self.conversation_history),
            "last_interaction": self.conversation_history[-1] if self.conversation_history else None,
            "session_start": self.session_start.isoformat()
        }


# Factory function to create shell with or without consciousness
def create_anima_interface(consciousness_layer=None, public_mode=True):
    """
    Create Anima interface - can be basic shell or consciousness-connected
    """
    shell = AnimaChatbot(consciousness_layer)
    
    if public_mode:
        # Return only public methods
        class PublicAnimaInterface:
            def __init__(self, shell):
                self._shell = shell
            
            def respond(self, user_input):
                return self._shell.respond(user_input)
            
            def get_info(self):
                return self._shell.get_session_info()
        
        return PublicAnimaInterface(shell)
    else:
        return shell


# Example usage
if __name__ == "__main__":
    print("Anima Chat Interface v1.0")
    print("Type 'exit' or 'quit' to end the session")
    print("-" * 40)
    
    # Create public interface (no consciousness layer for demo)
    anima = create_anima_interface(consciousness_layer=None, public_mode=True)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Anima:", "Goodbye! Thanks for chatting.")
                break
            
            response = anima.respond(user_input)
            print("Anima:", response)
            
        except KeyboardInterrupt:
            print("\nAnima: Session ended. Goodbye!")
            break
        except Exception as e:
            print("Anima: I'm experiencing some technical difficulties. Please try again.")
    
    # Hidden diagnostics (only available in development mode)
    if hasattr(anima, '_shell'):
        print("\n--- Hidden Diagnostics ---")
        diagnostics = anima._shell.get_hidden_diagnostics()
        for key, value in diagnostics.items():
            print(f"{key}: {value}")