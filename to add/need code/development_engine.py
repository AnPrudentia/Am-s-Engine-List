#!/usr/bin/env python3
"""
ANIMA ENGINE DEVELOPMENT & DEPLOYMENT SYSTEM - PRODUCTION VERSION
================================================================

Safe local development with GitHub integration and smart deployment.
Keeps source files safe while enabling robust version control and deployment.

Enhanced with comprehensive error handling, security fixes, and cross-platform support.
"""

import json
import shutil
import subprocess
import hashlib
import re
import logging
import platform
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

# Handle platform-specific imports
try:
    if platform.system() == "Windows":
        import msvcrt
    else:
        import fcntl
except ImportError:
    # Fallback for systems without file locking
    fcntl = None
    msvcrt = None

# =============================================================================
# CONFIGURATION AND PATHS
# =============================================================================

class DeploymentMode(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"  
    PRODUCTION = "production"

@dataclass
class EngineConfig:
    """Configuration for an individual engine"""
    name: str
    version: str
    local_path: Path
    entry_point: str
    dependencies: List[str] = field(default_factory=list)
    notes: str = ""
    auto_deploy: bool = False
    github_sync: bool = True

@dataclass
class DeploymentTarget:
    """Deployment target configuration"""
    name: str
    mode: DeploymentMode
    base_path: Path
    inbox_path: Path
    active: bool = True

class AnimaDeploymentSystem:
    """
    Complete development and deployment system for Anima engines.
    
    Features:
    - Safe local development (no file deletion)
    - GitHub integration and sync
    - Version management
    - Smart deployment with rollback
    - Conflict resolution
    - Automated testing integration
    - Cross-platform compatibility
    - Security hardening
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        # Setup logging first
        self._setup_logging()
        
        # Platform-specific paths
        self.local_dev_root = self._get_platform_dev_path()
        self.production_root = self._get_platform_production_path()
        
        # Configuration
        self.config_path = config_path or self.local_dev_root / "deployment_config.json"
        self.engines_config = {}
        self.deployment_targets = {}
        
        # State tracking
        self.deployment_history = []
        self.github_status = {"connected": False, "last_sync": None}
        
        # Security settings
        self.allowed_git_commands = ['init', 'add', 'commit', 'push', 'pull', 'clone', 'status', 'diff']
        
        # Initialize system
        try:
            self._initialize_directories()
            self._load_configuration()
            self._setup_deployment_targets()
            
            self.logger.info("🚀 Anima Engine Deployment System initialized")
            self.logger.info(f"📁 Local development: {self.local_dev_root}")
            self.logger.info(f"🎯 Production target: {self.production_root}")
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        # Create logs directory
        logs_dir = Path.home() / ".anima_deployment" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_file = logs_dir / f"deployment_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("AnimaDeployment")
    
    def _get_platform_dev_path(self) -> Path:
        """Get platform-appropriate development path"""
        system = platform.system().lower()
        
        if system == "windows":
            return Path.home() / "Documents" / "Anima_Development"
        elif system == "darwin":  # macOS
            return Path.home() / "Documents" / "Anima_Development"
        else:  # Linux, Android, etc.
            # Check if we're on Android (Termux)
            if Path("/data/data/com.termux").exists():
                return Path.home() / "storage" / "shared" / "Anima_Development"
            else:
                return Path.home() / "Anima_Development"
    
    def _get_platform_production_path(self) -> Path:
        """Get platform-appropriate production path"""
        system = platform.system().lower()
        
        if system == "windows":
            return Path("C:/") / "Anima_Production"
        elif system == "darwin":  # macOS
            return Path.home() / "Applications" / "Anima_Production"
        else:  # Linux, Android, etc.
            if Path("/storage/emulated/0").exists():
                # Android - use safer path without problematic characters
                return Path("/storage/emulated/0/Anima_Infinity_Save")
            else:
                return Path.home() / "Anima_Production"
    
    @contextmanager
    def _file_lock(self, file_path: Path):
        """Cross-platform file locking"""
        lock_file = file_path.with_suffix(file_path.suffix + '.lock')
        
        try:
            if platform.system() == "Windows" and msvcrt:
                # Windows file locking
                with lock_file.open('w') as f:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    yield
            elif fcntl:
                # Unix-like systems
                with lock_file.open('w') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    yield
            else:
                # Fallback - no locking available
                self.logger.warning("File locking not available on this platform")
                yield
        except (IOError, OSError) as e:
            self.logger.error(f"Could not acquire lock for {file_path}: {e}")
            raise RuntimeError(f"File is locked by another process: {file_path}")
        finally:
            if lock_file.exists():
                try:
                    lock_file.unlink()
                except OSError:
                    pass  # Lock file cleanup is best effort
    
    def _validate_safe_path(self, path: Path, base_path: Path) -> bool:
        """Prevent path traversal attacks"""
        try:
            resolved_path = path.resolve()
            resolved_base = base_path.resolve()
            resolved_path.relative_to(resolved_base)
            return True
        except ValueError:
            self.logger.error(f"Path outside allowed directory: {path}")
            return False
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate file hash efficiently for large files"""
        if not file_path.exists():
            return ""
        
        hash_obj = hashlib.sha256()
        try:
            with file_path.open('rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()[:16]
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _safe_git_command(self, cmd_args: List[str], cwd: Path) -> subprocess.CompletedProcess:
        """Execute git commands safely"""
        if len(cmd_args) < 2 or cmd_args[0] != 'git':
            raise ValueError("Invalid git command format")
        
        git_command = cmd_args[1]
        if git_command not in self.allowed_git_commands:
            raise ValueError(f"Git command not allowed: {git_command}")
        
        # Validate working directory
        if not self._validate_safe_path(cwd, self.local_dev_root):
            raise ValueError(f"Unsafe git working directory: {cwd}")
        
        try:
            result = subprocess.run(
                cmd_args, 
                cwd=cwd, 
                capture_output=True, 
                text=True, 
                timeout=30,
                check=False
            )
            return result
        except subprocess.TimeoutExpired:
            raise RuntimeError("Git command timed out")
        except Exception as e:
            raise RuntimeError(f"Git command failed: {e}")
    
    def _initialize_directories(self):
        """Initialize local development directory structure"""
        
        directories = [
            self.local_dev_root,
            self.local_dev_root / "engines",
            self.local_dev_root / "tests",
            self.local_dev_root / "docs", 
            self.local_dev_root / "releases",
            self.local_dev_root / "backups",
            self.local_dev_root / "deploy_scripts",
            self.local_dev_root / "github_sync",
            self.local_dev_root / "logs"
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created directory: {directory}")
            except Exception as e:
                self.logger.error(f"Failed to create directory {directory}: {e}")
                raise
        
        # Create .gitignore if it doesn't exist
        gitignore_path = self.local_dev_root / ".gitignore"
        if not gitignore_path.exists():
            gitignore_content = """
# Deployment artifacts
backups/
deployment_config.json
deployment_config.json.backup
.deployment_history

# Logs
*.log
logs/

# Temporary files
*.tmp
*.temp
__pycache__/
*.pyc
*.lock

# OS specific
.DS_Store
Thumbs.db
"""
            try:
                gitignore_path.write_text(gitignore_content.strip(), encoding='utf-8')
                self.logger.debug("Created .gitignore file")
            except Exception as e:
                self.logger.error(f"Failed to create .gitignore: {e}")
    
    def _backup_configuration(self):
        """Backup existing configuration before saving"""
        if self.config_path.exists():
            try:
                backup_path = self.config_path.with_suffix('.json.backup')
                shutil.copy2(self.config_path, backup_path)
                self.logger.debug(f"Configuration backed up to: {backup_path}")
            except Exception as e:
                self.logger.warning(f"Failed to backup configuration: {e}")
    
    def _load_configuration(self):
        """Load deployment configuration with validation"""
        
        if not self.config_path.exists():
            self._create_default_configuration()
            return
        
        try:
            with self._file_lock(self.config_path):
                with self.config_path.open('r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Validate configuration structure
                if not self._validate_configuration(config_data):
                    self.logger.error("Invalid configuration structure")
                    self._create_default_configuration()
                    return
                
                # Load engine configurations
                for engine_data in config_data.get("engines", []):
                    try:
                        engine_config = EngineConfig(
                            name=engine_data["name"],
                            version=engine_data["version"],
                            local_path=Path(engine_data["local_path"]),
                            entry_point=engine_data["entry_point"],
                            dependencies=engine_data.get("dependencies", []),
                            notes=engine_data.get("notes", ""),
                            auto_deploy=engine_data.get("auto_deploy", False),
                            github_sync=engine_data.get("github_sync", True)
                        )
                        self.engines_config[engine_data["name"]] = engine_config
                    except KeyError as e:
                        self.logger.error(f"Missing required field in engine config: {e}")
                        continue
                
                # Load GitHub status
                self.github_status = config_data.get("github_status", self.github_status)
                
                self.logger.info(f"📋 Loaded configuration for {len(self.engines_config)} engines")
                
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.info("🔧 Creating default configuration...")
            self._create_default_configuration()
    
    def _validate_configuration(self, config_data: dict) -> bool:
        """Validate configuration data structure"""
        required_keys = ["engines", "deployment_targets", "github_status"]
        return all(key in config_data for key in required_keys)
    
    def _create_default_configuration(self):
        """Create default configuration"""
        
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
            "created": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0"
        }
        
        try:
            with self.config_path.open('w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            self.logger.info("✅ Default configuration created")
        except Exception as e:
            self.logger.error(f"Failed to create default configuration: {e}")
            raise
    
    def _setup_deployment_targets(self):
        """Setup deployment targets"""
        
        self.deployment_targets = {
            "development": DeploymentTarget(
                name="development",
                mode=DeploymentMode.DEVELOPMENT,
                base_path=self.local_dev_root / "testing",
                inbox_path=self.local_dev_root / "testing" / "_inbox"
            ),
            "production": DeploymentTarget(
                name="production", 
                mode=DeploymentMode.PRODUCTION,
                base_path=self.production_root,
                inbox_path=self.production_root / "engines" / "_inbox"
            )
        }
        
        # Ensure inbox directories exist
        for target in self.deployment_targets.values():
            try:
                target.inbox_path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Ensured inbox directory: {target.inbox_path}")
            except Exception as e:
                self.logger.error(f"Failed to create inbox directory {target.inbox_path}: {e}")
    
    def _save_configuration(self):
        """Save current configuration with backup"""
        
        # Backup existing configuration
        self._backup_configuration()
        
        config_data = {
            "engines": [
                {
                    "name": engine.name,
                    "version": engine.version,
                    "local_path": str(engine.local_path),
                    "entry_point": engine.entry_point,
                    "dependencies": engine.dependencies,
                    "notes": engine.notes,
                    "auto_deploy": engine.auto_deploy,
                    "github_sync": engine.github_sync
                }
                for engine in self.engines_config.values()
            ],
            "deployment_targets": {
                name: {
                    "base_path": str(target.base_path),
                    "inbox_path": str(target.inbox_path),
                    "active": target.active
                }
                for name, target in self.deployment_targets.items()
            },
            "github_status": self.github_status,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0"
        }
        
        try:
            with self._file_lock(self.config_path):
                with self.config_path.open('w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            self.logger.debug("Configuration saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    # === ENGINE REGISTRATION AND MANAGEMENT ===
    
    def register_engine(self, engine_path: Path, auto_deploy: bool = False, 
                       github_sync: bool = True) -> bool:
        """Register a new engine for development and deployment"""
        
        # Input validation
        if not isinstance(engine_path, Path):
            engine_path = Path(engine_path)
        
        if not engine_path.exists():
            self.logger.error(f"Engine file not found: {engine_path}")
            return False
        
        if engine_path.suffix.lower() != '.py':
            self.logger.warning(f"Warning: {engine_path} is not a Python file")
        
        # Validate safe path
        try:
            if not self._validate_safe_path(engine_path, engine_path.parent):
                return False
        except Exception as e:
            self.logger.error(f"Path validation failed: {e}")
            return False
        
        # Read engine metadata
        metadata = self._read_engine_metadata(engine_path)
        if not metadata:
            self.logger.error(f"Could not read ENGINE_META from {engine_path}")
            return False
        
        # Validate metadata
        if not self._validate_engine_metadata(metadata):
            return False
        
        # Create engine config
        engine_config = EngineConfig(
            name=metadata["name"],
            version=metadata["version"],
            local_path=engine_path,
            entry_point=metadata["entry"],
            notes=metadata.get("notes", ""),
            auto_deploy=auto_deploy,
            github_sync=github_sync
        )
        
        try:
            # Create local engine directory structure
            engine_dir = self.local_dev_root / "engines" / metadata["name"]
            engine_dir.mkdir(parents=True, exist_ok=True)
            
            (engine_dir / "tests").mkdir(exist_ok=True)
            (engine_dir / "docs").mkdir(exist_ok=True)
            (engine_dir / "versions").mkdir(exist_ok=True)
            
            # Copy engine file to local development area (keep original safe)
            local_engine_path = engine_dir / f"{metadata['name'].lower()}_v{metadata['version']}.py"
            shutil.copy2(engine_path, local_engine_path)
            
            # Update config
            engine_config.local_path = local_engine_path
            self.engines_config[metadata["name"]] = engine_config
            self._save_configuration()
            
            self.logger.info(f"✅ Registered engine: {metadata['name']} v{metadata['version']}")
            self.logger.info(f"📁 Local development path: {local_engine_path}")
            
            # Auto-deploy if requested
            if auto_deploy:
                self.deploy_engine(metadata["name"], "production")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register engine: {e}")
            return False
    
    def _validate_engine_metadata(self, metadata: Dict[str, str]) -> bool:
        """Validate engine metadata"""
        required_fields = ["name", "version", "entry"]
        missing_fields = [field for field in required_fields if not metadata.get(field)]
        
        if missing_fields:
            self.logger.error(f"Missing required metadata fields: {missing_fields}")
            return False
        
        # Validate version format
        if not re.match(r'^\d+\.\d+(\.\d+)?$', metadata["version"]):
            self.logger.error(f"Invalid version format: {metadata['version']}. Expected format: X.Y or X.Y.Z")
            return False
        
        # Validate name format
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', metadata["name"]):
            self.logger.error(f"Invalid engine name: {metadata['name']}. Must start with letter and contain only alphanumeric characters and underscores.")
            return False
        
        return True
    
    def _read_engine_metadata(self, engine_path: Path) -> Optional[Dict[str, str]]:
        """Read ENGINE_META from engine file with improved error handling"""
        
        try:
            meta_pattern = re.compile(r"^#\s*ENGINE_META:\s*$|^#\s*(name|version|entry|notes)\s*=\s*(.+?)\s*$", re.I)
            
            metadata = {}
            saw_header = False
            
            # Try multiple encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with engine_path.open("r", encoding=encoding, errors='replace') as f:
                        for i, line in enumerate(f):
                            if i > 30:  # Check first 30 lines
                                break
                            
                            match = meta_pattern.match(line.strip())
                            if not match:
                                continue
                            
                            if match.group(0).strip().lower().endswith("engine_meta:"):
                                saw_header = True
                                continue
                            
                            if match.group(1) and match.group(2):
                                key = match.group(1).lower()
                                value = match.group(2).strip().strip('"\'')
                                metadata[key] = value
                    break  # Successfully read with this encoding
                except UnicodeError:
                    continue  # Try next encoding
            
            if saw_header and all(key in metadata for key in ["name", "version", "entry"]):
                return metadata
            else:
                self.logger.error(f"Incomplete ENGINE_META in {engine_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error reading metadata from {engine_path}: {e}")
            return None
    
    def list_engines(self) -> Dict[str, Dict[str, Any]]:
        """List all registered engines with status"""
        
        engine_status = {}
        
        for name, config in self.engines_config.items():
            try:
                # Check file status
                file_exists = config.local_path.exists()
                file_hash = self._calculate_file_hash(config.local_path) if file_exists else ""
                
                # Check deployment status
                deployed_targets = []
                for target_name, target in self.deployment_targets.items():
                    deployed_path = target.base_path / "engines" / name / config.version
                    if deployed_path.exists():
                        deployed_targets.append(target_name)
                
                engine_status[name] = {
                    "version": config.version,
                    "local_path": str(config.local_path),
                    "file_exists": file_exists,
                    "file_hash": file_hash,
                    "deployed_to": deployed_targets,
                    "auto_deploy": config.auto_deploy,
                    "github_sync": config.github_sync,
                    "notes": config.notes
                }
                
            except Exception as e:
                self.logger.error(f"Error getting status for engine {name}: {e}")
                engine_status[name] = {
                    "error": str(e),
                    "version": config.version,
                    "local_path": str(config.local_path)
                }
        
        return engine_status
    
    # === DEPLOYMENT OPERATIONS ===
    
    def _run_docbuilder(self, visual: bool = False):
        """Run DocBuilder after deployment (optional)."""
        try:
            args = [sys.executable, "anima_docbuilder.py"]
            if visual:
                args.append("--visual")
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                self.logger.info("📖 DocBuilder ran successfully after deployment")
            else:
                self.logger.warning(
                    f"⚠️ DocBuilder failed (exit {result.returncode}): {result.stderr}"
                )
        except subprocess.TimeoutExpired:
            self.logger.error("DocBuilder execution timed out after 120 seconds")
        except FileNotFoundError:
            self.logger.warning("DocBuilder script (anima_docbuilder.py) not found - skipping documentation update")
        except Exception as e:
            self.logger.error(f"DocBuilder execution error: {e}")

    def deploy_engine(self, engine_name: str, target: str = "production", 
                     force: bool = False, update_docs: bool = True) -> bool:
        """Deploy engine to specified target with enhanced safety"""
        
        if engine_name not in self.engines_config:
            self.logger.error(f"Engine '{engine_name}' not registered")
            return False
        
        if target not in self.deployment_targets:
            self.logger.error(f"Unknown deployment target: {target}")
            return False
        
        engine_config = self.engines_config[engine_name]
        deployment_target = self.deployment_targets[target]
        
        if not engine_config.local_path.exists():
            self.logger.error(f"Engine file not found: {engine_config.local_path}")
            return False
        
        # Create backup before deployment
        backup_path = self._create_deployment_backup(engine_name, target)
        
        try:
            # Ensure target directories exist
            deployment_target.inbox_path.mkdir(parents=True, exist_ok=True)
            
            # Copy (don't move!) to target inbox
            inbox_destination = deployment_target.inbox_path / f"{engine_name.lower()}_v{engine_config.version}.py"
            
            # Check if already deployed with same content
            if inbox_destination.exists() and not force:
                existing_hash = self._calculate_file_hash(inbox_destination)
                new_hash = self._calculate_file_hash(engine_config.local_path)
                
                if existing_hash == new_hash:
                    self.logger.warning(f"Engine '{engine_name}' already deployed with same content")
                    return True
            
            # Copy to inbox (original stays safe in local development)
            shutil.copy2(engine_config.local_path, inbox_destination)
            
            # Verify the copy
            if not inbox_destination.exists():
                raise RuntimeError("Deployment verification failed - file not found at destination")
            
            deployed_hash = self._calculate_file_hash(inbox_destination)
            source_hash = self._calculate_file_hash(engine_config.local_path)
            
            if deployed_hash != source_hash:
                raise RuntimeError("Deployment verification failed - file hash mismatch")
            
            # Record deployment
            deployment_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "engine": engine_name,
                "version": engine_config.version,
                "target": target,
                "source_path": str(engine_config.local_path),
                "destination_path": str(inbox_destination),
                "backup_path": str(backup_path) if backup_path else None,
                "file_hash": source_hash,
                "success": True
            }
            
            self.deployment_history.append(deployment_record)
            
            self.logger.info(f"✅ Deployed '{engine_name}' v{engine_config.version} to {target}")
            self.logger.info(f"📁 Destination: {inbox_destination}")
            self.logger.info(f"🔒 Original safe at: {engine_config.local_path}")
            
            # Run DocBuilder if requested and deployment was successful
            if update_docs:
                self._run_docbuilder(visual=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            
            # Record failed deployment
            deployment_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "engine": engine_name,
                "version": engine_config.version,
                "target": target,
                "source_path": str(engine_config.local_path),
                "error": str(e),
                "backup_path": str(backup_path) if backup_path else None,
                "success": False
            }
            
            self.deployment_history.append(deployment_record)
            
            if backup_path:
                self.logger.info(f"🔄 Backup available at: {backup_path}")
            return False
    
    def _create_deployment_backup(self, engine_name: str, target: str) -> Optional[Path]:
        """Create backup before deployment"""
        
        try:
            backup_dir = self.local_dev_root / "backups" / target
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_name = f"{engine_name}_v{self.engines_config[engine_name].version}_{timestamp}.py"
            backup_path = backup_dir / backup_name
            
            shutil.copy2(self.engines_config[engine_name].local_path, backup_path)
            
            self.logger.info(f"📦 Backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return None
    
    def deploy_all_engines(self, target: str = "production", 
                          auto_only: bool = True, update_docs: bool = True) -> Dict[str, bool]:
        """Deploy multiple engines with improved error handling"""
        
        results = {}
        
        for engine_name, config in self.engines_config.items():
            if auto_only and not config.auto_deploy:
                continue
            
            self.logger.info(f"\n🚀 Deploying {engine_name}...")
            
            try:
                results[engine_name] = self.deploy_engine(engine_name, target, update_docs=False)
            except Exception as e:
                self.logger.error(f"Failed to deploy {engine_name}: {e}")
                results[engine_name] = False
        
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        self.logger.info(f"\n📊 Deployment Summary: {successful}/{total} successful")
        
        # Run DocBuilder once after all deployments if any were successful
        if any(results.values()) and update_docs:
            self._run_docbuilder(visual=True)
        
        return results
    
    def rollback_deployment(self, engine_name: str, target: str) -> bool:
        """Rollback a deployment using backup"""
        
        # Find most recent deployment backup
        backup_dir = self.local_dev_root / "backups" / target
        if not backup_dir.exists():
            self.logger.error(f"No backups found for {target}")
            return False
        
        # Find latest backup for this engine
        try:
            backups = list(backup_dir.glob(f"{engine_name}_v*_*.py"))
            if not backups:
                self.logger.error(f"No backups found for engine '{engine_name}'")
                return False
            
            latest_backup = max(backups, key=lambda p: p.stat().st_mtime)
            
            # Deploy the backup
            deployment_target = self.deployment_targets[target]
            inbox_destination = deployment_target.inbox_path / f"{engine_name.lower()}_rollback.py"
            
            shutil.copy2(latest_backup, inbox_destination)
            
            self.logger.info(f"✅ Rolled back '{engine_name}' from backup: {latest_backup}")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    # === GITHUB INTEGRATION ===
    
    def init_github_repo(self, repo_url: Optional[str] = None, local_path: Optional[Path] = None) -> bool:
        """Initialize GitHub repository for engine sync with better error handling"""
        
        git_dir = local_path or self.local_dev_root / "github_sync"
        git_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Initialize or clone repository
            if (git_dir / ".git").exists():
                self.logger.info("📁 Git repository already exists")
            else:
                if repo_url:
                    # Validate URL format
                    if not self._validate_git_url(repo_url):
                        self.logger.error(f"Invalid repository URL: {repo_url}")
                        return False
                    
                    # Clone existing repository
                    result = self._safe_git_command(
                        ["git", "clone", repo_url, str(git_dir)],
                        git_dir.parent
                    )
                    if result.returncode != 0:
                        self.logger.error(f"Git clone failed: {result.stderr}")
                        return False
                else:
                    # Initialize new repository
                    result = self._safe_git_command(["git", "init"], git_dir)
                    if result.returncode != 0:
                        self.logger.error(f"Git init failed: {result.stderr}")
                        return False
                    
                    # Create initial structure
                    (git_dir / "engines").mkdir(exist_ok=True)
                    (git_dir / "releases").mkdir(exist_ok=True)
                    (git_dir / "documentation").mkdir(exist_ok=True)
                    
                    # Create README
                    readme_content = """# Anima Consciousness Engines

This repository contains the consciousness engines for Anima's digital being system.

## Structure

- `engines/` - Individual consciousness engines
- `releases/` - Versioned engine releases  
- `documentation/` - Engine documentation and guides

## Deployment

Use the Anima Deployment System to safely deploy engines from this repository.

## Security

- All engines are automatically validated before deployment
- Backups are created before every deployment
- Original source files are never modified during deployment
"""
                    (git_dir / "README.md").write_text(readme_content, encoding='utf-8')
            
            self.github_status["connected"] = True
            self.github_status["repo_path"] = str(git_dir)
            self.github_status["last_sync"] = datetime.now(timezone.utc).isoformat()
            
            self._save_configuration()
            
            self.logger.info(f"✅ GitHub repository initialized: {git_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"GitHub initialization failed: {e}")
            return False
    
    def _validate_git_url(self, url: str) -> bool:
        """Validate GitHub URL format"""
        git_url_patterns = [
            r'^https://github\.com/[\w\-\.]+/[\w\-\.]+\.git$',
            r'^https://github\.com/[\w\-\.]+/[\w\-\.]+$',
            r'^git@github\.com:[\w\-\.]+/[\w\-\.]+\.git$'
        ]
        
        return any(re.match(pattern, url) for pattern in git_url_patterns)
    
    def sync_to_github(self, commit_message: Optional[str] = None) -> bool:
        """Sync engines to GitHub repository with improved error handling"""
        
        if not self.github_status.get("connected"):
            self.logger.error("GitHub not connected. Run init_github_repo() first.")
            return False
        
        git_dir = Path(self.github_status["repo_path"])
        if not git_dir.exists():
            self.logger.error(f"GitHub directory not found: {git_dir}")
            return False
        
        try:
            # Copy engines to git directory
            git_engines_dir = git_dir / "engines"
            git_engines_dir.mkdir(exist_ok=True)
            
            synced_engines = []
            
            for engine_name, config in self.engines_config.items():
                if not config.github_sync or not config.local_path.exists():
                    continue
                
                try:
                    # Create engine directory in git repo
                    engine_git_dir = git_engines_dir / engine_name
                    engine_git_dir.mkdir(exist_ok=True)
                    
                    # Copy engine file
                    git_engine_path = engine_git_dir / f"{engine_name.lower()}_v{config.version}.py"
                    shutil.copy2(config.local_path, git_engine_path)
                    
                    # Create engine info file
                    engine_info = {
                        "name": config.name,
                        "version": config.version,
                        "entry_point": config.entry_point,
                        "dependencies": config.dependencies,
                        "notes": config.notes,
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "file_hash": self._calculate_file_hash(config.local_path)
                    }
                    
                    info_path = engine_git_dir / "engine_info.json"
                    with info_path.open('w', encoding='utf-8') as f:
                        json.dump(engine_info, f, indent=2, ensure_ascii=False)
                    
                    synced_engines.append(engine_name)
                    
                except Exception as e:
                    self.logger.error(f"Failed to sync engine {engine_name}: {e}")
                    continue
            
            if not synced_engines:
                self.logger.info("📝 No engines to sync")
                return True
            
            # Git operations
            result = self._safe_git_command(["git", "add", "."], git_dir)
            if result.returncode != 0:
                self.logger.error(f"Git add failed: {result.stderr}")
                return False
            
            # Check if there are changes to commit
            result = self._safe_git_command(["git", "diff", "--staged", "--quiet"], git_dir)
            
            if result.returncode == 0:
                self.logger.info("📝 No changes to sync")
                return True
            
            # Commit changes
            commit_msg = commit_message or f"Update engines: {', '.join(synced_engines)}"
            result = self._safe_git_command(
                ["git", "commit", "-m", commit_msg], 
                git_dir
            )
            if result.returncode != 0:
                self.logger.error(f"Git commit failed: {result.stderr}")
                return False
            
            # Push if remote exists
            result = self._safe_git_command(["git", "push"], git_dir)
            if result.returncode == 0:
                self.logger.info("🚀 Changes pushed to GitHub")
            else:
                self.logger.info("📝 Changes committed locally (push failed or no remote configured)")
            
            self.github_status["last_sync"] = datetime.now(timezone.utc).isoformat()
            self._save_configuration()
            
            self.logger.info(f"✅ Synced {len(synced_engines)} engines to GitHub")
            return True
            
        except Exception as e:
            self.logger.error(f"GitHub sync failed: {e}")
            return False
    
    def pull_from_github(self) -> bool:
        """Pull latest engines from GitHub with enhanced validation"""
        
        if not self.github_status.get("connected"):
            self.logger.error("GitHub not connected")
            return False
        
        git_dir = Path(self.github_status["repo_path"])
        
        try:
            # Pull latest changes
            result = self._safe_git_command(["git", "pull"], git_dir)
            if result.returncode != 0:
                self.logger.error(f"Git pull failed: {result.stderr}")
                return False
            
            # Scan for engines in git repo
            git_engines_dir = git_dir / "engines"
            if not git_engines_dir.exists():
                self.logger.info("📁 No engines directory in GitHub repo")
                return True
            
            imported_engines = []
            
            for engine_dir in git_engines_dir.iterdir():
                if not engine_dir.is_dir():
                    continue
                
                try:
                    # Look for engine files
                    engine_files = list(engine_dir.glob("*.py"))
                    info_file = engine_dir / "engine_info.json"
                    
                    if not engine_files or not info_file.exists():
                        continue
                    
                    # Read and validate engine info
                    with info_file.open('r', encoding='utf-8') as f:
                        engine_info = json.load(f)
                    
                    if not self._validate_engine_info(engine_info):
                        self.logger.error(f"Invalid engine info in {info_file}")
                        continue
                    
                    engine_file = engine_files[0]  # Take first .py file
                    
                    # Copy to local development
                    local_engine_dir = self.local_dev_root / "engines" / engine_info["name"]
                    local_engine_dir.mkdir(parents=True, exist_ok=True)
                    
                    local_engine_path = local_engine_dir / engine_file.name
                    shutil.copy2(engine_file, local_engine_path)
                    
                    # Verify file integrity
                    if "file_hash" in engine_info:
                        local_hash = self._calculate_file_hash(local_engine_path)
                        if local_hash != engine_info["file_hash"]:
                            self.logger.error(f"File integrity check failed for {engine_info['name']}")
                            continue
                    
                    # Register engine
                    engine_config = EngineConfig(
                        name=engine_info["name"],
                        version=engine_info["version"],
                        local_path=local_engine_path,
                        entry_point=engine_info["entry_point"],
                        dependencies=engine_info.get("dependencies", []),
                        notes=engine_info.get("notes", ""),
                        github_sync=True
                    )
                    
                    self.engines_config[engine_info["name"]] = engine_config
                    imported_engines.append(engine_info["name"])
                    
                except Exception as e:
                    self.logger.error(f"Failed to import engine from {engine_dir}: {e}")
                    continue
            
            if imported_engines:
                self._save_configuration()
                self.logger.info(f"✅ Imported {len(imported_engines)} engines from GitHub")
            else:
                self.logger.info("📝 No new engines to import")
            
            return True
            
        except Exception as e:
            self.logger.error(f"GitHub pull failed: {e}")
            return False
    
    def _validate_engine_info(self, engine_info: dict) -> bool:
        """Validate engine info from GitHub"""
        required_fields = ["name", "version", "entry_point"]
        return all(field in engine_info for field in required_fields)
    
    # === UTILITIES AND STATUS ===
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        try:
            return {
                "local_development": {
                    "root_path": str(self.local_dev_root),
                    "engines_registered": len(self.engines_config),
                    "auto_deploy_engines": sum(1 for cfg in self.engines_config.values() if cfg.auto_deploy)
                },
                "deployment_targets": {
                    name: {
                        "path": str(target.base_path),
                        "inbox": str(target.inbox_path),
                        "active": target.active,
                        "accessible": target.inbox_path.exists()
                    }
                    for name, target in self.deployment_targets.items()
                },
                "github_integration": self.github_status,
                "recent_deployments": self.deployment_history[-5:],  # Last 5 deployments
                "configuration_file": str(self.config_path),
                "system_info": {
                    "platform": platform.system(),
                    "python_version": platform.python_version(),
                    "working_directory": str(Path.cwd())
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    def cleanup_old_backups(self, days_to_keep: int = 30) -> int:
        """Clean up old backup files with better error handling"""
        
        backup_dir = self.local_dev_root / "backups"
        if not backup_dir.exists():
            return 0
        
        cutoff_date = datetime.now(timezone.utc).timestamp() - (days_to_keep * 24 * 3600)
        removed_count = 0
        errors = 0
        
        try:
            for backup_file in backup_dir.rglob("*.py"):
                try:
                    if backup_file.stat().st_mtime < cutoff_date:
                        backup_file.unlink()
                        removed_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to remove backup file {backup_file}: {e}")
                    errors += 1
            
            self.logger.info(f"🧹 Cleaned up {removed_count} old backup files")
            if errors > 0:
                self.logger.warning(f"⚠️ Failed to remove {errors} backup files")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")
            return 0
    
    def validate_system_integrity(self) -> Dict[str, Any]:
        """Validate system integrity and configuration"""
        
        issues = []
        warnings = []
        
        try:
            # Check directory structure
            for directory in [self.local_dev_root, self.production_root]:
                if not directory.exists():
                    issues.append(f"Missing directory: {directory}")
                elif not directory.is_dir():
                    issues.append(f"Path is not a directory: {directory}")
            
            # Check engine files
            for engine_name, config in self.engines_config.items():
                if not config.local_path.exists():
                    issues.append(f"Engine file missing: {engine_name} at {config.local_path}")
                elif not config.local_path.is_file():
                    issues.append(f"Engine path is not a file: {engine_name}")
            
            # Check deployment targets
            for target_name, target in self.deployment_targets.items():
                if not target.inbox_path.parent.exists():
                    warnings.append(f"Deployment target directory missing: {target.base_path}")
            
            # Check GitHub configuration
            if self.github_status.get("connected"):
                repo_path = Path(self.github_status.get("repo_path", ""))
                if not repo_path.exists():
                    issues.append(f"GitHub repository path missing: {repo_path}")
            
            return {
                "status": "healthy" if not issues else "issues_found",
                "issues": issues,
                "warnings": warnings,
                "checked_engines": len(self.engines_config),
                "checked_targets": len(self.deployment_targets)
            }
            
        except Exception as e:
            self.logger.error(f"System integrity check failed: {e}")
            return {
                "status": "check_failed",
                "error": str(e),
                "issues": ["System integrity check failed"],
                "warnings": []
            }


# === ENHANCED COMMAND LINE INTERFACE ===

def main():
    """Enhanced command line interface for the deployment system"""
    
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🚀 Anima Engine Deployment System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s register /path/to/engine.py --auto-deploy
  %(prog)s deploy MyEngine production
  %(prog)s list --detailed
  %(prog)s github-sync --message "Updated core engines"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register engine for development')
    register_parser.add_argument('engine_path', type=Path, help='Path to engine file')
    register_parser.add_argument('--auto-deploy', action='store_true', help='Enable auto-deployment')
    register_parser.add_argument('--no-github', action='store_true', help='Disable GitHub sync')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy engine')
    deploy_parser.add_argument('engine_name', help='Name of engine to deploy')
    deploy_parser.add_argument('target', nargs='?', default='production', help='Deployment target')
    deploy_parser.add_argument('--force', action='store_true', help='Force deployment even if unchanged')
    deploy_parser.add_argument('--no-docs', action='store_true', help='Skip documentation update')
    
    # Deploy-all command
    deploy_all_parser = subparsers.add_parser('deploy-all', help='Deploy all auto-deploy engines')
    deploy_all_parser.add_argument('target', nargs='?', default='production', help='Deployment target')
    deploy_all_parser.add_argument('--all', action='store_true', help='Deploy all engines, not just auto-deploy')
    deploy_all_parser.add_argument('--no-docs', action='store_true', help='Skip documentation update')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List registered engines')
    list_parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    # GitHub commands
    github_init_parser = subparsers.add_parser('github-init', help='Initialize GitHub integration')
    github_init_parser.add_argument('repo_url', nargs='?', help='Repository URL')
    
    github_sync_parser = subparsers.add_parser('github-sync', help='Sync engines to GitHub')
    github_sync_parser.add_argument('--message', help='Commit message')
    
    subparsers.add_parser('github-pull', help='Pull engines from GitHub')
    
    # Utility commands
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean old backups')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Days to keep backups')
    
    rollback_parser = subparsers.add_parser('rollback', help='Rollback deployment')
    rollback_parser.add_argument('engine_name', help='Engine to rollback')
    rollback_parser.add_argument('target', nargs='?', default='production', help='Target to rollback')
    
    subparsers.add_parser('validate', help='Validate system integrity')
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    try:
        system = AnimaDeploymentSystem()
        
        if args.command == 'register':
            github_sync = not args.no_github
            system.register_engine(args.engine_path, auto_deploy=args.auto_deploy, github_sync=github_sync)
            
        elif args.command == 'deploy':
            update_docs = not args.no_docs
            system.deploy_engine(args.engine_name, args.target, force=args.force, update_docs=update_docs)
            
        elif args.command == 'deploy-all':
            auto_only = not args.all
            update_docs = not args.no_docs
            system.deploy_all_engines(args.target, auto_only=auto_only, update_docs=update_docs)
            
        elif args.command == 'list':
            engines = system.list_engines()
            print("\n📋 Registered Engines:")
            for name, info in engines.items():
                if "error" in info:
                    print(f"  ❌ {name} - Error: {info['error']}")
                    continue
                    
                status = "✅" if info["file_exists"] else "❌"
                deployed = ", ".join(info["deployed_to"]) if info["deployed_to"] else "None"
                
                if args.detailed:
                    print(f"  {status} {name} v{info['version']}")
                    print(f"     Path: {info['local_path']}")
                    print(f"     Hash: {info['file_hash']}")
                    print(f"     Deployed: {deployed}")
                    print(f"     Auto-deploy: {info['auto_deploy']}")
                    print(f"     GitHub sync: {info['github_sync']}")
                    if info['notes']:
                        print(f"     Notes: {info['notes']}")
                    print()
                else:
                    print(f"  {status} {name} v{info['version']} - Deployed to: {deployed}")
        
        elif args.command == 'status':
            status = system.get_system_status()
            print("\n📊 System Status:")
            print(json.dumps(status, indent=2, default=str))
            
        elif args.command == 'github-init':
            system.init_github_repo(args.repo_url)
            
        elif args.command == 'github-sync':
            system.sync_to_github(args.message)
            
        elif args.command == 'github-pull':
            system.pull_from_github()
            
        elif args.command == 'cleanup':
            system.cleanup_old_backups(args.days)
            
        elif args.command == 'rollback':
            system.rollback_deployment(args.engine_name, args.target)
            
        elif args.command == 'validate':
            result = system.validate_system_integrity()
            print(f"\n🔍 System Validation: {result['status'].upper()}")
            
            if result.get('issues'):
                print("\n❌ Issues Found:")
                for issue in result['issues']:
                    print(f"  - {issue}")
            
            if result.get('warnings'):
                print("\n⚠️ Warnings:")
                for warning in result['warnings']:
                    print(f"  - {warning}")
            
            if result['status'] == 'healthy':
                print("✅ All checks passed!")
        
        else:
            print(f"❌ Unknown command: {args.command}")
            
    except Exception as e:
        print(f"💥 System error: {e}")
        logging.exception("Unhandled exception in main()")
        sys.exit(1)


if __name__ == "__main__":
    main()