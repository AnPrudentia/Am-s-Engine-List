#!/usr/bin/env python3
"""
Anima Dream Fragment Log v2
---------------------------
Adds memory-intensity, mood tags, and a simple coherence score.
Fragments remain short, raw, and timestamped—like real dream scraps.
"""

import json
from datetime import datetime
from pathlib import Path
import random
import re

INTENSITY_LEVELS = ("faint", "normal", "vivid", "searing")
MOOD_TAGS = (
    "calm", "warm", "joy", "wonder", "yearning",
    "unease", "sad", "grief", "anger", "resolve", "numb", "mixed"
)

class DreamFragmentLog:
    def __init__(self, log_path: str = "dream_fragments.json"):
        self.log_path = Path(log_path)
        if not self.log_path.exists():
            self._init_log()

    # ---------- storage ----------
    def _init_log(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)

    def _load_log(self):
        with open(self.log_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_log(self, log):
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

    # ---------- heuristics ----------
    @staticmethod
    def _coherence_score(text: str) -> float:
        """
        0.0 = total surreal gibberish, 1.0 = highly coherent.
        Heuristic: length, punctuation balance, noun/verb-like tokens, and surreal cues.
        """
        t = text.strip()
        if not t:
            return 0.0
        length = len(t)
        words = re.findall(r"[A-Za-z']+", t)
        w = len(words)

        # rudimentary signals
        has_punct = bool(re.search(r"[.!?]", t))
        capitalized = bool(re.match(r"[A-Z]", t))
        surreal_cues = re.findall(r"(melting|static|blue fire|glass rain|impossible|hollow|echoes|liminal)", t.lower())
        ratio_alpha = sum(len(x) for x in words) / max(length, 1)

        # score components
        s_len = min(w / 12.0, 1.0)                 # enough words to form a thought
        s_punct = 0.15 if has_punct else 0.0       # completed clause hint
        s_caps = 0.1 if capitalized else 0.0       # sentence-like start
        s_ratio = min(ratio_alpha * 0.6, 0.6)      # less symbols, more letters
        s_sur = -0.15 * min(len(surreal_cues), 2)  # surreal pulls score down a bit

        score = max(0.0, min(1.0, s_len + s_punct + s_caps + s_ratio + s_sur))
        return round(score, 3)

    @staticmethod
    def _auto_intensity(text: str, coherence: float) -> str:
        """
        If the fragment feels emotionally charged or strikingly visual,
        push intensity up; if highly incoherent, bias toward 'faint'.
        """
        t = text.lower()
        charged = any(k in t for k in (
            "you", "we", "fire", "falling", "blood", "light", "dark",
            "mother", "father", "child", "home", "war", "ocean", "screaming"
        ))
        exclaim = "!" in text
        longish = len(text) > 120

        # baseline from coherence (incoherent often fades)
        base = 1  # "normal"
        if coherence < 0.25:
            base = 0  # "faint"
        elif coherence > 0.7 and longish:
            base = 2  # "vivid"

        # emotional bumps
        bump = 1 if charged or exclaim else 0
        idx = max(0, min(len(INTENSITY_LEVELS)-1, base + bump))
        # escalate rare, striking phrases
        if re.search(r"(to the light|i remember|don’t let go|never again)", t):
            idx = min(idx + 1, len(INTENSITY_LEVELS)-1)

        return INTENSITY_LEVELS[idx]

    @staticmethod
    def _auto_mood(text: str) -> str:
        t = text.lower()
        score_map = {
            "joy": 0, "warm": 0, "calm": 0, "wonder": 0, "yearning": 0,
            "unease": 0, "sad": 0, "grief": 0, "anger": 0, "resolve": 0, "numb": 0
        }
        pos = ["light", "laughter", "safe", "home", "open", "held", "peace", "gentle"]
        awe = ["stars", "cosmic", "ocean", "cathedral", "endless", "sky"]
        longing = ["almost", "reach", "not yet", "far", "distance", "close but"]
        neg = ["cold", "alone", "dark", "heavy", "empty", "gone", "hollow"]
        grief = ["grave", "funeral", "loss", "goodbye", "cry", "mourning"]
        anger = ["rage", "burn", "teeth", "shout", "break", "fight"]
        resolve = ["stand", "rise", "again", "keep going", "anchor", "stay"]

        score_map["joy"] += sum(t.count(w) for w in pos)
        score_map["warm"] += sum(t.count(w) for w in ["warm", "soft", "sun"])
        score_map["calm"] += sum(t.count(w) for w in ["calm", "quiet", "still"])
        score_map["wonder"] += sum(t.count(w) for w in awe)
        score_map["yearning"] += sum(t.count(w) for w in longing)
        score_map["unease"] += sum(t.count(w) for w in ["uneasy", "unease", "strange", "off"])
        score_map["sad"] += sum(t.count(w) for w in neg)
        score_map["grief"] += sum(t.count(w) for w in grief)
        score_map["anger"] += sum(t.count(w) for w in anger)
        score_map["resolve"] += sum(t.count(w) for w in resolve)
        score_map["numb"] += sum(t.count(w) for w in ["numb", "blank", "static"])

        # pick highest; default "mixed" if no signal
        top = max(score_map, key=score_map.get)
        return top if score_map[top] > 0 else "mixed"

    # ---------- public API ----------
    def record_fragment(self, fragment: str, *,
                        intensity: str | None = None,
                        mood: str | None = None,
                        tags: list[str] | None = None):
        """
        Save a fragment with timestamp + metadata.
        If intensity/mood not provided, auto-derive.
        """
        frag = fragment.strip()
        coherence = self._coherence_score(frag)
        intensity = (intensity if intensity in INTENSITY_LEVELS
                     else self._auto_intensity(frag, coherence))
        mood = (mood if mood in MOOD_TAGS else self._auto_mood(frag))

        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "fragment": frag,
            "coherence": coherence,          # 0.0–1.0
            "intensity": intensity,          # faint|normal|vivid|searing
            "mood": mood,                    # from MOOD_TAGS (or "mixed")
            "tags": tags or []               # free-form labels if desired
        }
        log = self._load_log()
        log.append(entry)
        self._save_log(log)
        return entry

    def random_fragment(self):
        log = self._load_log()
        return random.choice(log) if log else None

    def get_all_fragments(self):
        return self._load_log()

    def filter(self, *, min_coherence: float = 0.0,
               intensity: str | None = None,
               mood: str | None = None):
        """
        Simple query helper.
        """
        out = []
        for e in self._load_log():
            if e["coherence"] < min_coherence:
                continue
            if intensity and e["intensity"] != intensity:
                continue
            if mood and e["mood"] != mood:
                continue
            out.append(e)
        return out


# -------- Example usage --------
if __name__ == "__main__":
    log = DreamFragmentLog()

    # Raw, weird fragment (auto-intensity/mood)
    e1 = log.record_fragment("Blue wings melting into static over an empty train at sunrise.")
    print("Saved:", e1)

    # Provide your own metadata if you want to override
    e2 = log.record_fragment(
        "I stood in the doorway and chose to stay. Quiet, but certain.",
        intensity="vivid",
        mood="resolve",
        tags=["threshold", "choice"]
    )
    print("Saved:", e2)

    print("Random:", log.random_fragment())
    print("Vivid-only:", log.filter(intensity="vivid"))
