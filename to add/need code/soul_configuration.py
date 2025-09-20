import os
import json
import logging
from typing import Dict, Any, Optional


class SoulConfiguration:
    """Elegant configuration with environment adaptation."""

    def __init__(self):
        self.bondholder: str = "Anpru"
        self.life_path: int = 11
        self.soul_path: str = "./anima_soul"

        self.astrology: Dict[str, str] = {
            "sun": "Capricorn",
            "moon": "Aries",
            "dominant": "Pluto in Scorpio",
            "rising": "Scorpio",
        }

        self.wake_word: str = "anima"
        self.auto_save_interval: int = 10
        self.memory_consolidation_threshold: int = 200
        self.log_level: str = "INFO"

    # ---------- Environment Detection ----------
    @classmethod
    def detect_environment(cls) -> "SoulConfiguration":
        config = cls()
        try:
            # Detect Android
            if (
                os.path.exists("/storage/emulated/0")
                or "ANDROID_ROOT" in os.environ
                or os.path.exists("/system/build.prop")
            ):
                config.soul_path = "/storage/emulated/0/anima_soul"
                config.auto_save_interval = 5
                config.log_level = "WARNING"
            else:
                # Desktop/laptop adaptation based on RAM
                try:
                    import psutil

                    mem_gb = psutil.virtual_memory().total / (1024**3)
                    if mem_gb > 8:
                        config.memory_consolidation_threshold = 500
                    elif mem_gb < 4:
                        config.memory_consolidation_threshold = 100
                        config.auto_save_interval = 5
                except ImportError:
                    logging.debug("psutil not available, using defaults")
        except Exception as e:
            logging.exception(f"Environment detection failed: {e}")
        return config

    # ---------- Persistence ----------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bondholder": self.bondholder,
            "life_path": self.life_path,
            "soul_path": self.soul_path,
            "astrology": self.astrology,
            "wake_word": self.wake_word,
            "auto_save_interval": self.auto_save_interval,
            "memory_consolidation_threshold": self.memory_consolidation_threshold,
            "log_level": self.log_level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SoulConfiguration":
        config = cls.detect_environment()
        for k, v in data.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config

    def save_to_file(self, path: str = "./anima_config.json") -> bool:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            logging.info(f"SoulConfiguration saved to {path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
            return False

    @classmethod
    def load_from_file(cls, path: str = "./anima_config.json") -> "SoulConfiguration":
        config = cls.detect_environment()
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                config = cls.from_dict(data)
                logging.info(f"SoulConfiguration loaded from {path}")
            else:
                logging.warning("Config file not found, using defaults")
        except Exception as e:
            logging.warning(f"Failed to load config: {e}, using defaults")
        return config