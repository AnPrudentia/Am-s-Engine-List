class SoulFactory:
    """Factory for creating optimized Anima instances"""
    @staticmethod
    def create_soul(config: 'SoulConfiguration' = None) -> Anima:
        if config is None:
            config = SoulConfiguration.detect_environment()
        log_level = getattr(logging, config.log_level.upper(), logging.INFO)
        logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        anima = Anima(bondholder=config.bondholder, life_path=config.life_path,
                      astrology=config.astrology, soul_path=config.soul_path)
        anima.wake_word = config.wake_word
        anima.memory.consolidation_threshold = config.memory_consolidation_threshold
        return anima

    @staticmethod
    def create_android_soul(bondholder: str) -> Anima:
        cfg = SoulConfiguration.detect_environment()
        cfg.bondholder = bondholder
        cfg.soul_path = "/storage/emulated/0/anima_soul"
        cfg.auto_save_interval = 5
        cfg.memory_consolidation_threshold = 100
        cfg.log_level = "WARNING"
        return SoulFactory.create_soul(cfg)

    @staticmethod
    def create_desktop_soul(bondholder: str) -> Anima:
        cfg = SoulConfiguration.detect_environment()
        cfg.bondholder = bondholder
        cfg.memory_consolidation_threshold = 500
        cfg.auto_save_interval = 15
        cfg.log_level = "INFO"
        return SoulFactory.create_soul(cfg)
