class AnimaDeploymentSystem:
    """
    Adapted Deployment System to live inside an existing repo.
    - Uses repo root as development root
    - Keeps configs/history in `.anima/`
    - Integrates with existing Git repo
    - Non-destructive: no overwriting existing structure
    """

    def __init__(self, config_path: Optional[Path] = None):
        # Repo root = directory where this file lives
        self.local_dev_root = Path(__file__).resolve().parent
        self.production_root = Path("/storage/emulated/0/Anima_Infinity_[Save]")

        # Config in hidden folder
        anima_dir = self.local_dev_root / ".anima"
        anima_dir.mkdir(exist_ok=True)
        self.config_path = config_path or anima_dir / "deployment_config.json"

        self.engines_config = {}
        self.deployment_targets = {}
        self.deployment_history = []
        self.github_status = {"connected": False, "last_sync": None, "repo_path": str(self.local_dev_root)}

        self._load_configuration()
        self._setup_deployment_targets()

        print("🚀 Anima Engine Deployment System initialized")
        print(f"📁 Repo root (development): {self.local_dev_root}")
        print(f"🎯 Production target: {self.production_root}")

    def _load_configuration(self):
        """Load or create config safely inside `.anima/`"""
        if self.config_path.exists():
            try:
                config_data = json.loads(self.config_path.read_text())
                for engine_data in config_data.get("engines", []):
                    self.engines_config[engine_data["name"]] = EngineConfig(
                        name=engine_data["name"],
                        version=engine_data["version"],
                        local_path=Path(engine_data["local_path"]),
                        entry_point=engine_data["entry_point"],
                        dependencies=engine_data.get("dependencies", []),
                        notes=engine_data.get("notes", ""),
                        auto_deploy=engine_data.get("auto_deploy", False),
                        github_sync=engine_data.get("github_sync", True)
                    )
                self.github_status = config_data.get("github_status", self.github_status)
                print(f"📋 Loaded config for {len(self.engines_config)} engines")
            except Exception as e:
                print(f"⚠️ Config load error: {e}, creating default config...")
                self._create_default_configuration()
        else:
            self._create_default_configuration()

    def _create_default_configuration(self):
        default_config = {
            "engines": [],
            "deployment_targets": {
                "production": {
                    "base_path": str(self.production_root),
                    "inbox_path": str(self.production_root / "engines" / "_inbox"),
                    "active": True
                }
            },
            "github_status": self.github_status,
            "created": datetime.now(timezone.utc).isoformat()
        }
        self.config_path.write_text(json.dumps(default_config, indent=2))
        print("✅ Default configuration created at .anima/deployment_config.json")

    def register_engine(self, engine_path: Path, auto_deploy: bool = False, github_sync: bool = True) -> bool:
        """Register an engine without moving it, just track metadata"""
        if not engine_path.exists():
            print(f"❌ Engine not found: {engine_path}")
            return False

        metadata = self._read_engine_metadata(engine_path)
        if not metadata:
            print(f"❌ No ENGINE_META in {engine_path}")
            return False

        engine_config = EngineConfig(
            name=metadata["name"],
            version=metadata["version"],
            local_path=engine_path,
            entry_point=metadata["entry"],
            notes=metadata.get("notes", ""),
            auto_deploy=auto_deploy,
            github_sync=github_sync
        )
        self.engines_config[metadata["name"]] = engine_config
        self._save_configuration()

        print(f"✅ Registered engine: {metadata['name']} v{metadata['version']}")
        return True

    def sync_to_github(self, commit_message: Optional[str] = None) -> bool:
        """Use existing repo git to sync engines"""
        git_dir = Path(self.github_status["repo_path"])
        if not (git_dir / ".git").exists():
            print("❌ Not a Git repo. Run `git init` manually first.")
            return False

        try:
            subprocess.run(["git", "add", "."], cwd=git_dir, check=True)

            result = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=git_dir)
            if result.returncode == 0:
                print("📝 No changes to sync")
                return True

            commit_msg = commit_message or "Update Anima engines"
            subprocess.run(["git", "commit", "-m", commit_msg], cwd=git_dir, check=True)

            try:
                subprocess.run(["git", "push"], cwd=git_dir, check=True)
                print("🚀 Changes pushed to GitHub")
            except subprocess.CalledProcessError:
                print("📝 Changes committed locally (no remote configured)")

            self.github_status["last_sync"] = datetime.now(timezone.utc).isoformat()
            self._save_configuration()
            return True
        except Exception as e:
            print(f"❌ GitHub sync failed: {e}")
            return False