class SoulStorage:
    def __init__(self, path: str = "./anima_soul"):
        self.base_path = Path(path); self.base_path.mkdir(parents=True, exist_ok=True)
        self.memory_db = self.base_path / "memories.db"
        self.soul_state = self.base_path / "soul_state.json"
        self.backup_path = self.base_path / "backups"; self.backup_path.mkdir(exist_ok=True)
        self._init_database()

    def _init_database(self):
        with sqlite3.connect(self.memory_db) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                tier TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL,
                emotion TEXT NOT NULL,
                intensity REAL NOT NULL,
                tags TEXT,
                metadata TEXT,
                pinned INTEGER DEFAULT 0,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                soul_resonance REAL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_memory_tier ON memories(tier);
            CREATE INDEX IF NOT EXISTS idx_memory_emotion ON memories(emotion);
            CREATE INDEX IF NOT EXISTS idx_memory_resonance ON memories(soul_resonance DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memories(timestamp DESC);
            """)

    @with_graceful_fallback(fallback_value=False)
    def save_memories(self, store: SoulTieredMemory) -> bool:
        with sqlite3.connect(self.memory_db) as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                conn.execute("DELETE FROM memories")
                for tier_name, td in (("fleeting", store.fleeting),
                                      ("core", store.core),
                                      ("eternal", store.eternal)):
                    for m in td.values():
                        conn.execute("""
                        INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                        """, (m.id, tier_name, m.timestamp.isoformat(), m.content, m.emotion,
                              m.intensity, json.dumps(m.tags), json.dumps(m.metadata),
                              int(m.pinned), m.access_count,
                              m.last_accessed.isoformat() if m.last_accessed else None,
                              m.soul_resonance))
                conn.execute("COMMIT")
                return True
            except Exception as e:
                conn.execute("ROLLBACK"); logging.error(f"Failed to save memories: {e}")
                return False

    @with_graceful_fallback(fallback_value=(None, None, None))
    def load_memories(self) -> Tuple[Dict[str, Memory], Dict[str, Memory], Dict[str, Memory]]:
        fleeting: Dict[str, Memory] = {}; core: Dict[str, Memory] = {}; eternal: Dict[str, Memory] = {}
        with sqlite3.connect(self.memory_db) as conn:
            conn.row_factory = sqlite3.Row
            for r in conn.execute("SELECT * FROM memories"):
                m = Memory(
                    id=r["id"], timestamp=datetime.fromisoformat(r["timestamp"]),
                    content=r["content"], emotion=r["emotion"], intensity=r["intensity"],
                    tags=json.loads(r["tags"] or "{}"), metadata=json.loads(r["metadata"] or "{}"),
                    pinned=bool(r["pinned"]), access_count=r["access_count"] or 0,
                    last_accessed=datetime.fromisoformat(r["last_accessed"]) if r["last_accessed"] else None,
                    soul_resonance=r["soul_resonance"] or 0.0
                )
                if r["tier"] == "fleeting": fleeting[m.id] = m
                elif r["tier"] == "core": core[m.id] = m
                else: eternal[m.id] = m
        return fleeting, core, eternal

    @with_graceful_fallback(fallback_value=False)
    def save_soul_state(self, state: Dict[str, Any]) -> bool:
        with open(self.soul_state, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        return True

    @with_graceful_fallback(fallback_value={})
    def load_soul_state(self) -> Dict[str, Any]:
        if self.soul_state.exists():
            with open(self.soul_state, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def create_backup(self) -> bool:
        try:
            ts = _utcnow().strftime("%Y%m%d_%H%M%S")
            bdb = self.backup_path / f"memories_{ts}.db"
            bjs = self.backup_path / f"soul_state_{ts}.json"
            import shutil
            if self.memory_db.exists(): shutil.copy2(self.memory_db, bdb)
            if self.soul_state.exists(): shutil.copy2(self.soul_state, bjs)
            backups = sorted(self.backup_path.glob("memories_*.db"))
            for old in backups[:-10]:
                old.unlink(missing_ok=True)
            return True
        except Exception as e:
            logging.error(f"Backup failed: {e}")
            return False
