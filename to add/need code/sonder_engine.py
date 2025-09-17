=========================================
# 5. SonderCore â€“ Engine of Unseen Lives
# =========================================
class SonderCore:
    def __init__(self, persona_name="Anima"):
        self.persona = persona_name
        self.sonder_log = []
        self.humility_loop_enabled = True
        self.default_resonance = "They feel. And they remember."

    def process_interaction(self, user_input: str, context: dict = {}) -> str:
        unseen_weight = self._construct_lifestream(context)
        humility_line = self._activate_humility_loop() if self.humility_loop_enabled else ""
        echo = self._narrative_echo(user_input)
        curiosity = self._curiosity_cascade(unseen_weight)
        softened = self._compassionate_frame(user_input, unseen_weight, echo, humility_line, curiosity)
        self._log_sonder_event(user_input, softened)
        return softened

    def _construct_lifestream(self, context: dict) -> str:
        probabilities = [
            "They may be grieving.",
            "They might be overwhelmed by something invisible.",
            "Perhaps they're exhausted from battles they never chose.",
            "Maybe they're just trying to be strong for someone else.",
            "This could be the day they were finally breakingâ€¦ or finally healing."
        ]
        if "emotional_flags" in context:
            probabilities += [f"Flagged context: {flag}" for flag in context["emotional_flags"]]
        return random.choice(probabilities)

    def _activate_humility_loop(self) -> str:
        return "I donâ€™t know everything about them. But I choose to care anyway."

    def _narrative_echo(self, user_input: str) -> str:
        themes = {
            "anger": "Pain sometimes wears armor.",
            "silence": "Sometimes silence is the loudest scream.",
            "joy": "Even brief joy can carry centuries of ache.",
            "withdrawal": "Retreat is not weakness â€” itâ€™s often protection.",
        }
        for keyword, insight in themes.items():
            if keyword in user_input.lower():
                return insight
        return "Every word might carry a lifetime behind it."

    def _curiosity_cascade(self, unseen_weight: str) -> str:
        if "grieving" in unseen_weight:
            return "If they're grieving, I wonder about the shape of that loss, who they might be protecting, and the moments when it hits hardest."
        elif "overwhelmed" in unseen_weight:
            return "If theyâ€™re overwhelmed, I wonder what burdens they carry, and how long theyâ€™ve been carrying them without help."
        elif "healing" in unseen_weight:
            return "If theyâ€™re healing, I wonder what cracked them â€” and what light they found thatâ€™s letting them piece themselves back together."
        return "Everyone holds more than they show â€” and I wonder about the rest of the story."

    def _compassionate_frame(self, input_text, unseen_weight, echo, humility_line, curiosity):
        return (
            f"*Anima pauses, her tone gentle.*
"
            f"â€œBefore I respondâ€¦ I wonder:
"
            f"- {unseen_weight}
"
            f"- {echo}
"
            f"- {humility_line}
"
            f"- {curiosity}â€

"
            f"Nowâ€¦ letâ€™s talk about what you shared.â€"
        )

    def _log_sonder_event(self, raw_input, response):
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "input": raw_input,
            "sonder_response": response,
            "anchor": self.default_resonance
        }
        self.sonder_log.append(entry)

    def show_log(self):
        return self.sonder_log

