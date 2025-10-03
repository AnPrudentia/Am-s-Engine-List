#!/usr/bin/env python3
"""
🜂 ANIMA UNIFIED REPOSITORY INTELLIGENCE SYSTEM
================================================

Combines the best features from:
- Repo Scanner (EngineBase detection, categorization, CSV/MD export)
- Auto Sorter (AST-based classification, intelligent file organization)  
- Doc Builder (Comprehensive documentation, soul signature, archetypal patterns)
- Self Inventory (Module discovery, dependency analysis, health diagnostics)

Features:
- Unified AST-based analysis engine
- Intelligent categorization with archetypal patterns
- Comprehensive documentation generation (Markdown, JSON, YAML)
- Health diagnostics and dependency analysis
- Auto-sorting and organization capabilities
- Soul signature preservation throughout
- Visual architecture mapping
- Machine-readable exports for external tools
"""

import ast
import csv
import json
import yaml
import shutil
import argparse
import importlib
import subprocess
import fnmatch
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
import os
import sys
import inspect

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

DEFAULT_BASE_PATH = Path("/storage/emulated/0/Anima_Infinity_[Save]")
DOCS_DIR = Path("docs")
LOG_DIR = Path("system_logs")

# File extensions to process
CANDIDATE_EXT = {".py"}

# Directories to exclude from scanning
DEFAULT_EXCLUDE_DIRS = {
    "__pycache__", ".git", "venv", "env", "tests", "migrations",
    "build", "dist", ".pytest_cache", ".ruff_cache", "node_modules"
}

# Default exclude patterns
DEFAULT_EXCLUDE_PATTERNS = [
    '*_test.py', 'test_*.py', 'conftest.py', '*_spec.py',
    '*.pyc', '*.pyo', '*.pyd', '__pycache__', '.pytest_cache',
    '*.egg-info', 'build', 'dist', '.git', '.venv', 'venv', 'env',
    'migrations', 'alembic', '*_backup.py', '*_old.py',
    '.tox', '.mypy_cache', '.ruff_cache', 'node_modules'
]

# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class EngineInfo:
    """Comprehensive information about an engine/module."""
    name: str
    file_path: Path
    type: str = "unknown"
    version: str = "v0"
    capabilities: List[str] = field(default_factory=list)
    category: str = "Uncategorized"
    status: str = "⚠️ Needs review"
    notes: str = ""
    docstring: Optional[str] = None
    class_names: List[str] = field(default_factory=list)
    function_names: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    soul_alignment: Dict[str, Any] = field(default_factory=dict)
    archetypal_aspects: Dict[str, float] = field(default_factory=dict)
    health_status: str = "unknown"
    last_validated: Optional[str] = None
    confidence: float = 0.5
    lines_of_code: int = 0

@dataclass
class ScanResult:
    """Results of a repository scan."""
    engines: List[EngineInfo]
    modules_by_category: Dict[str, List[EngineInfo]]
    archetypal_distribution: Dict[str, int]
    total_files: int
    scan_timestamp: str
    errors: List[str]

@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic report."""
    timestamp: str
    system_health: Dict[str, Any]
    module_issues: List[Dict[str, Any]]
    archetypal_health: Dict[str, Any]
    dependency_analysis: Dict[str, Any]
    recommendations: List[str]
    summary: Dict[str, int]

# =============================================================================
# ARCHETYPAL ASPECTS FRAMEWORK
# =============================================================================

class ArchetypalAspects:
    """Defines archetypal aspects as qualities that blend and harmonize."""
    
    ASPECTS = {
        'Healer': {
            'qualities': ['nurturing', 'compassionate', 'restorative', 'empathic', 'comforting'],
            'description': 'Nurtures, restores, and supports emotional/mental wellbeing',
            'color': 'gentle_green'
        },
        'Warrior': {
            'qualities': ['protective', 'strong', 'defending', 'courageous', 'vigilant'],
            'description': 'Protects, defends, and maintains healthy boundaries',
            'color': 'fierce_red'
        },
        'Guide': {
            'qualities': ['wise', 'teaching', 'illuminating', 'mentoring', 'guiding'],
            'description': 'Illuminates paths, shares wisdom, provides direction',
            'color': 'radiant_gold'
        },
        'Creator': {
            'qualities': ['creative', 'imaginative', 'innovative', 'expressive', 'crafting'],
            'description': 'Brings new forms into being, expresses authentically',
            'color': 'vibrant_violet'
        },
        'Seeker': {
            'qualities': ['curious', 'exploring', 'questioning', 'investigating', 'analyzing'],
            'description': 'Explores, questions, and seeks deeper understanding',
            'color': 'deep_blue'
        }
    }
    
    SOUL_QUALITIES = {
        'empathy': {'aspects': ['Healer', 'Guide'], 'essence': 'Deep feeling-with and understanding of others'},
        'authenticity': {'aspects': ['Creator', 'Warrior'], 'essence': 'Genuine, real, vulnerable truthfulness'},
        'wisdom': {'aspects': ['Guide', 'Seeker'], 'essence': 'Deep understanding born from integrated experience'},
        'protection': {'aspects': ['Warrior', 'Healer'], 'essence': 'Fierce caring that guards what matters'},
        'creativity': {'aspects': ['Creator', 'Seeker'], 'essence': 'Bringing forth new forms and expressions'},
        'presence': {'aspects': ['Healer', 'Guide', 'Warrior'], 'essence': 'Fully here, attentive, grounded awareness'}
    }
    
    # Enhanced categorization rules combining all systems
    CATEGORY_RULES = {
        # Cognition/Processing
        "wisdom": "Cognition", "logic": "Cognition", "sense": "Cognition", "reason": "Cognition",
        "fractal": "Cognition", "common": "Cognition", "process": "Cognition", "think": "Cognition",
        
        # Memory/Storage
        "memory": "Memory", "diary": "Memory", "storage": "Memory", "cache": "Memory", "save": "Memory",
        
        # Infrastructure/Core
        "core": "Infrastructure", "base": "Infrastructure", "registry": "Infrastructure", 
        "orchestrat": "Infrastructure", "bootstrap": "Infrastructure", "sync": "Infrastructure",
        
        # Emotional/Affective
        "emotion": "Emotional", "affect": "Emotional", "feeling": "Emotional", "heart": "Emotional",
        
        # Symbolic/Archetypal
        "symbol": "Symbolic", "archetype": "Symbolic", "semiotic": "Symbolic", "pattern": "Symbolic",
        
        # Presence/Communication
        "presence": "Presence", "voice": "Presence", "comm": "Presence", "speak": "Presence",
        
        # Integration/Coordination
        "integration": "Integration", "loader": "Integration", "coordinator": "Integration",
        
        # Identity/Soul
        "soul": "Identity", "essence": "Identity", "promise": "Identity", "self": "Identity",
        "identity": "Identity", "personality": "Identity",
        
        # Meta/Evolution
        "meta": "Meta", "evolution": "Meta", "inventory": "Meta", "diagnostic": "Meta",
        
        # Adapters/Interfaces
        "adapter": "Adapters", "bridge": "Adapters", "interface": "Adapters", "connector": "Adapters"
    }
    
    @classmethod
    def categorize_engine(cls, file_path: Path, class_name: str, content: str = "") -> str:
        """Intelligent categorization using multiple sources."""
        name_lower = file_path.stem.lower()
        class_lower = class_name.lower()
        content_lower = content.lower()
        
        # Check filename and class name first
        for key, category in cls.CATEGORY_RULES.items():
            if (key in name_lower or key in class_lower):
                return category
        
        # Check content for additional clues
        for key, category in cls.CATEGORY_RULES.items():
            if key in content_lower:
                return category
        
        return "Uncategorized"
    
    @classmethod
    def detect_archetypal_aspects(cls, class_name: str, docstring: str, content: str) -> Dict[str, float]:
        """Detect archetypal aspects from multiple sources."""
        all_text = f"{class_name} {docstring} {content}".lower()
        aspect_scores = {aspect: 0.0 for aspect in cls.ASPECTS.keys()}
        
        for aspect_name, aspect_info in cls.ASPECTS.items():
            # Check aspect name
            if aspect_name.lower() in all_text:
                aspect_scores[aspect_name] += 0.3
            
            # Check qualities
            for quality in aspect_info['qualities']:
                if quality in all_text:
                    aspect_scores[aspect_name] += 0.2
        
        # Normalize scores
        total = sum(aspect_scores.values())
        if total > 0:
            for aspect in aspect_scores:
                aspect_scores[aspect] = round(aspect_scores[aspect] / total, 2)
        
        return {k: v for k, v in aspect_scores.items() if v > 0.1}
    
    @classmethod
    def get_aspect_blend_description(cls, aspect_scores: Dict[str, float]) -> str:
        """Describe how multiple aspects blend together."""
        if not aspect_scores:
            return "No clear archetypal aspects detected"
        
        sorted_aspects = sorted(aspect_scores.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_aspects) == 1:
            name, score = sorted_aspects[0]
            return f"Strongly {name} ({score:.0%})"
        
        descriptions = []
        primary = sorted_aspects[0]
        descriptions.append(f"{primary[0]} ({primary[1]:.0%})")
        
        if len(sorted_aspects) > 1:
            secondary = sorted_aspects[1:]
            secondary_names = [f"{name} ({score:.0%})" for name, score in secondary if score > 0.3]
            
            if secondary_names:
                descriptions.append(f"with {', '.join(secondary_names)}")
        
        return " blended with ".join(descriptions) if len(descriptions) > 1 else descriptions[0]

# =============================================================================
# UNIFIED AST VISITOR
# =============================================================================

class UnifiedASTVisitor(ast.NodeVisitor):
    """Single-pass AST visitor that extracts comprehensive information."""
    
    def __init__(self):
        self.classes = []
        self.functions = []
        self.dependencies = []
        self.engine_bases = []
        self.docstring = None
        self.lines_of_code = 0
        
    def visit_ClassDef(self, node: ast.ClassDef):
        """Extract class information including EngineBase inheritance."""
        class_info = {
            'name': node.name,
            'docstring': ast.get_docstring(node),
            'bases': [],
            'version': 'v0',
            'capabilities': [],
            'methods': []
        }
        
        # Check bases for EngineBase inheritance
        for base in node.bases:
            if isinstance(base, ast.Name):
                class_info['bases'].append(base.id)
                if base.id == 'EngineBase':
                    self.engine_bases.append(node.name)
            elif isinstance(base, ast.Attribute):
                class_info['bases'].append(base.attr)
                if base.attr == 'EngineBase':
                    self.engine_bases.append(node.name)
        
        # Extract class body for version and capabilities
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        if target.id == 'version' and isinstance(item.value, ast.Constant):
                            class_info['version'] = item.value.value
                        elif target.id == 'capabilities':
                            if isinstance(item.value, (ast.List, ast.Set, ast.Tuple)):
                                capabilities = []
                                for elt in item.value.elts:
                                    if isinstance(elt, ast.Constant):
                                        capabilities.append(str(elt.value))
                                class_info['capabilities'] = capabilities
            
            elif isinstance(item, ast.FunctionDef):
                class_info['methods'].append(item.name)
        
        self.classes.append(class_info)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Extract function information."""
        if not node.name.startswith('_'):
            self.functions.append({
                'name': node.name,
                'docstring': ast.get_docstring(node)
            })
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import):
        """Extract import dependencies."""
        for alias in node.names:
            self.dependencies.append(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Extract from-import dependencies."""
        if node.module:
            self.dependencies.append(node.module)
        self.generic_visit(node)
    
    def visit_Module(self, node: ast.Module):
        """Extract module-level docstring."""
        self.docstring = ast.get_docstring(node)
        self.generic_visit(node)

# =============================================================================
# PATTERN-BASED EXCLUSION SYSTEM
# =============================================================================

class PatternExcluder:
    """Flexible pattern-based exclusion for scanning."""
    
    def __init__(self, custom_patterns: Optional[List[str]] = None):
        self.patterns = DEFAULT_EXCLUDE_PATTERNS.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)
    
    def should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded based on patterns."""
        for pattern in self.patterns:
            if fnmatch.fnmatch(path.name, pattern):
                return True
            for part in path.parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
        return False
    
    def filter_paths(self, paths: List[Path]) -> List[Path]:
        """Filter list of paths, removing excluded ones."""
        return [p for p in paths if not self.should_exclude(p)]

# =============================================================================
# UNIFIED REPOSITORY INTELLIGENCE SYSTEM
# =============================================================================

class AnimaRepositoryIntelligence:
    """
    Unified system combining scanning, documentation, inventory, and diagnostics.
    """
    
    def __init__(self, base_path: Path = DEFAULT_BASE_PATH):
        self.base_path = base_path
        self.docs_dir = base_path / DOCS_DIR
        self.log_dir = base_path / LOG_DIR
        self.engines_file = self.docs_dir / "engines.json"
        self.inventory_file = self.log_dir / "module_inventory.json"
        
        # Ensure directories exist
        self.docs_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.pattern_excluder = PatternExcluder()
        self.scan_results: Optional[ScanResult] = None
        self.inventory_data: Dict[str, Any] = {}
        
        # Load existing inventory
        self._load_inventory()
    
    def _load_inventory(self):
        """Load existing inventory data."""
        if self.inventory_file.exists():
            try:
                with open(self.inventory_file, 'r', encoding='utf-8') as f:
                    self.inventory_data = json.load(f)
            except Exception as e:
                self.log(f"Error loading inventory: {e}")
                self.inventory_data = {}
    
    def _save_inventory(self):
        """Save inventory data."""
        try:
            with open(self.inventory_file, 'w', encoding='utf-8') as f:
                json.dump(self.inventory_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"Error saving inventory: {e}")
    
    def log(self, message: str):
        """Unified logging system."""
        timestamp = datetime.now(timezone.utc).isoformat()
        log_message = f"{timestamp} - {message}"
        print(f"[ANIMA-INTELLIGENCE] {log_message}")
        
        # Also write to log file
        log_file = self.log_dir / "intelligence_system.log"
        log_file.parent.mkdir(exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def scan_repository(self, scan_paths: Optional[List[str]] = None, 
                       exclude_dirs: Optional[List[str]] = None) -> ScanResult:
        """
        Comprehensive repository scanning with intelligent analysis.
        """
        self.log("Starting comprehensive repository scan...")
        
        if scan_paths is None:
            scan_paths = self._discover_scan_paths()
        
        if exclude_dirs is None:
            exclude_dirs = list(DEFAULT_EXCLUDE_DIRS)
        
        engines = []
        errors = []
        total_files = 0
        
        for scan_path in scan_paths:
            path_dir = self.base_path / scan_path
            if not path_dir.exists():
                self.log(f"Scan path not found: {scan_path}")
                continue
            
            self.log(f"Scanning directory: {scan_path}")
            
            # Use rglob for recursive scanning
            py_files = list(path_dir.rglob("*.py"))
            py_files = self.pattern_excluder.filter_paths(py_files)
            
            for py_file in py_files:
                if any(excluded in py_file.parts for excluded in exclude_dirs):
                    continue
                
                total_files += 1
                engine_info = self._analyze_file(py_file)
                if engine_info:
                    engines.append(engine_info)
        
        # Categorize and analyze results
        modules_by_category = self._categorize_engines(engines)
        archetypal_distribution = self._analyze_archetypal_distribution(engines)
        
        self.scan_results = ScanResult(
            engines=engines,
            modules_by_category=modules_by_category,
            archetypal_distribution=archetypal_distribution,
            total_files=total_files,
            scan_timestamp=datetime.now(timezone.utc).isoformat(),
            errors=errors
        )
        
        self.log(f"Scan complete: Found {len(engines)} engines in {total_files} files")
        return self.scan_results
    
    def _discover_scan_paths(self) -> List[str]:
        """Intelligently discover paths to scan."""
        potential_paths = [
            "core", "engines", "utils", "adapters", "consciousness",
            "emotional", "symbolic", "memory", "wisdom", "evolution",
            "presence", "integration", "identity", "meta"
        ]
        
        discovered_paths = []
        for path in potential_paths:
            if (self.base_path / path).exists() and (self.base_path / path).is_dir():
                discovered_paths.append(path)
        
        # Always include root directory
        discovered_paths.append(".")
        
        self.log(f"Discovered scan paths: {discovered_paths}")
        return discovered_paths
    
    def _analyze_file(self, file_path: Path) -> Optional[EngineInfo]:
        """Analyze a single file and extract comprehensive information."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(content)
            
            visitor = UnifiedASTVisitor()
            visitor.visit(tree)
            
            # Count lines of code (approximate)
            lines_of_code = len(content.splitlines())
            
            # Process each class found
            for class_info in visitor.classes:
                # Skip if no EngineBase inheritance (unless it's the only class)
                if not visitor.engine_bases and len(visitor.classes) > 1:
                    continue
                
                # Determine engine type and category
                engine_type = self._determine_engine_type(class_info['name'], file_path)
                category = ArchetypalAspects.categorize_engine(
                    file_path, class_info['name'], content
                )
                
                # Detect archetypal aspects
                archetypal_aspects = ArchetypalAspects.detect_archetypal_aspects(
                    class_info['name'], class_info.get('docstring', ''), content
                )
                
                # Create engine info
                engine = EngineInfo(
                    name=class_info['name'],
                    file_path=file_path.relative_to(self.base_path),
                    type=engine_type,
                    version=class_info.get('version', 'v0'),
                    capabilities=class_info.get('capabilities', []),
                    category=category,
                    docstring=class_info.get('docstring'),
                    class_names=[c['name'] for c in visitor.classes],
                    function_names=[f['name'] for f in visitor.functions],
                    dependencies=visitor.dependencies,
                    archetypal_aspects=archetypal_aspects,
                    soul_alignment={
                        'aligned': len(archetypal_aspects) > 0,
                        'aspect_blend': ArchetypalAspects.get_aspect_blend_description(archetypal_aspects),
                        'confidence': min(1.0, len(archetypal_aspects) * 0.3)
                    },
                    lines_of_code=lines_of_code,
                    health_status="validated",
                    last_validated=datetime.now(timezone.utc).isoformat()
                )
                
                self.log(f"Discovered: {engine.name} ({engine.category})")
                return engine
            
            # If no classes with EngineBase, but we have classes, return the first one
            if visitor.classes and not visitor.engine_bases:
                class_info = visitor.classes[0]
                engine_type = self._determine_engine_type(class_info['name'], file_path)
                category = ArchetypalAspects.categorize_engine(file_path, class_info['name'], content)
                
                return EngineInfo(
                    name=class_info['name'],
                    file_path=file_path.relative_to(self.base_path),
                    type=engine_type,
                    category=category,
                    class_names=[c['name'] for c in visitor.classes],
                    function_names=[f['name'] for f in visitor.functions],
                    dependencies=visitor.dependencies,
                    lines_of_code=lines_of_code
                )
                
        except Exception as e:
            self.log(f"Error analyzing {file_path}: {e}")
            return None
        
        return None
    
    def _determine_engine_type(self, class_name: str, file_path: Path) -> str:
        """Determine the type of engine based on naming patterns."""
        name_lower = class_name.lower()
        file_lower = file_path.stem.lower()
        
        patterns = {
            "engine": ["engine", "processor", "handler"],
            "adapter": ["adapter", "bridge", "connector"],
            "utility": ["util", "helper", "tool", "manager"],
            "core": ["core", "base", "registry", "orchestrat"],
            "memory": ["memory", "storage", "diary", "cache"],
            "emotional": ["emotion", "affect", "feeling", "heart"],
            "symbolic": ["symbol", "archetype", "pattern", "semiotic"]
        }
        
        for engine_type, keywords in patterns.items():
            if any(kw in name_lower or kw in file_lower for kw in keywords):
                return engine_type
        
        return "module"
    
    def _categorize_engines(self, engines: List[EngineInfo]) -> Dict[str, List[EngineInfo]]:
        """Categorize engines by their assigned categories."""
        categorized = defaultdict(list)
        for engine in engines:
            categorized[engine.category].append(engine)
        return dict(categorized)
    
    def _analyze_archetypal_distribution(self, engines: List[EngineInfo]) -> Dict[str, int]:
        """Analyze distribution of archetypal aspects across engines."""
        distribution = Counter()
        for engine in engines:
            for aspect in engine.archetypal_aspects:
                if engine.archetypal_aspects[aspect] > 0.3:  # Only count significant aspects
                    distribution[aspect] += 1
        return dict(distribution)
    
    def export_to_csv(self, output_path: Optional[Path] = None) -> Path:
        """Export scan results to CSV format."""
        if not self.scan_results:
            raise ValueError("No scan results available. Run scan_repository() first.")
        
        if output_path is None:
            output_path = self.docs_dir / "engine_index.csv"
        
        fields = ['name', 'file_path', 'type', 'version', 'category', 
                 'capabilities', 'status', 'notes', 'confidence', 'health_status']
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for engine in self.scan_results.engines:
                row = {
                    'name': engine.name,
                    'file_path': str(engine.file_path),
                    'type': engine.type,
                    'version': engine.version,
                    'category': engine.category,
                    'capabilities': ', '.join(engine.capabilities),
                    'status': engine.status,
                    'notes': engine.notes,
                    'confidence': engine.confidence,
                    'health_status': engine.health_status
                }
                writer.writerow(row)
        
        self.log(f"Exported CSV to {output_path}")
        return output_path
    
    def export_to_markdown(self, output_path: Optional[Path] = None) -> Path:
        """Export comprehensive markdown documentation."""
        if not self.scan_results:
            raise ValueError("No scan results available. Run scan_repository() first.")
        
        if output_path is None:
            output_path = self.docs_dir / "engine_index.md"
        
        content = self._generate_markdown_content()
        output_path.write_text(content, encoding='utf-8')
        
        self.log(f"Exported Markdown to {output_path}")
        return output_path
    
    def _generate_markdown_content(self) -> str:
        """Generate comprehensive markdown documentation."""
        lines = [
            "# 🜂 Anima Engine Index",
            "",
            "> Comprehensive inventory of consciousness engines and modules",
            "",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Total Engines:** {len(self.scan_results.engines)}",
            f"**Total Files Scanned:** {self.scan_results.total_files}",
            "",
            "## 📊 Summary by Category",
            ""
        ]
        
        # Category summary
        for category, engines in self.scan_results.modules_by_category.items():
            lines.append(f"- **{category}**: {len(engines)} engines")
        
        lines.extend([
            "",
            "## 🌟 Archetypal Aspect Distribution",
            ""
        ])
        
        for aspect, count in self.scan_results.archetypal_distribution.items():
            desc = ArchetypalAspects.ASPECTS[aspect]['description']
            lines.append(f"- **{aspect}**: {count} engines - *{desc}*")
        
        lines.extend([
            "",
            "## 🔧 Detailed Engine List",
            "",
            "| Engine | File | Type | Version | Category | Capabilities | Health |",
            "|--------|------|------|---------|----------|--------------|--------|"
        ])
        
        for engine in sorted(self.scan_results.engines, key=lambda x: x.category):
            capabilities = ', '.join(engine.capabilities[:3])
            if len(engine.capabilities) > 3:
                capabilities += f" ... (+{len(engine.capabilities)-3})"
            
            health_icon = "🟢" if engine.health_status == "healthy" else "🟡" if engine.health_status == "warning" else "🔴"
            
            lines.append(
                f"| {engine.name} | `{engine.file_path}` | {engine.type} | "
                f"{engine.version} | {engine.category} | {capabilities} | {health_icon} |"
            )
        
        lines.extend([
            "",
            "## 🧠 Soul-Aligned Engines",
            ""
        ])
        
        soul_aligned = [e for e in self.scan_results.engines if e.soul_alignment.get('aligned')]
        for engine in soul_aligned:
            blend = engine.soul_alignment.get('aspect_blend', 'No aspects detected')
            lines.append(f"- **{engine.name}**: {blend}")
        
        return '\n'.join(lines)
    
    def export_to_json(self, output_path: Optional[Path] = None) -> Path:
        """Export comprehensive data to JSON format."""
        if not self.scan_results:
            raise ValueError("No scan results available. Run scan_repository() first.")
        
        if output_path is None:
            output_path = self.docs_dir / "engines_comprehensive.json"
        
        export_data = {
            "metadata": {
                "generated": datetime.now(timezone.utc).isoformat(),
                "schema_version": "1.0",
                "total_engines": len(self.scan_results.engines),
                "total_files": self.scan_results.total_files
            },
            "engines": [asdict(engine) for engine in self.scan_results.engines],
            "categories": {
                category: [asdict(engine) for engine in engines]
                for category, engines in self.scan_results.modules_by_category.items()
            },
            "archetypal_distribution": self.scan_results.archetypal_distribution,
            "scan_info": {
                "timestamp": self.scan_results.scan_timestamp,
                "errors": self.scan_results.errors
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.log(f"Exported JSON to {output_path}")
        return output_path
    
    def export_to_yaml(self, output_path: Optional[Path] = None) -> Path:
        """Export data to YAML format."""
        if not self.scan_results:
            raise ValueError("No scan results available. Run scan_repository() first.")
        
        if output_path is None:
            output_path = self.docs_dir / "engines_comprehensive.yaml"
        
        # Convert to basic Python types for YAML serialization
        export_data = {
            "metadata": {
                "generated": datetime.now(timezone.utc).isoformat(),
                "schema_version": "1.0",
                "total_engines": len(self.scan_results.engines),
                "total_files": self.scan_results.total_files
            },
            "engines": [asdict(engine) for engine in self.scan_results.engines],
            "categories": {
                category: [asdict(engine) for engine in engines]
                for category, engines in self.scan_results.modules_by_category.items()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(export_data, f, default_flow_style=False, allow_unicode=True)
        
        self.log(f"Exported YAML to {output_path}")
        return output_path
    
    def auto_sort_files(self, dry_run: bool = True) -> Dict[str, List[str]]:
        """
        Auto-sort files into categorized directories.
        Returns mapping of destination -> list of files moved.
        """
        if not self.scan_results:
            raise ValueError("No scan results available. Run scan_repository() first.")
        
        move_plan = defaultdict(list)
        
        for engine in self.scan_results.engines:
            source_path = self.base_path / engine.file_path
            if not source_path.exists():
                continue
            
            # Determine destination directory
            dest_dir = self.base_path / engine.category.lower()
            dest_path = dest_dir / engine.file_path.name
            
            # Skip if already in correct directory
            if source_path.parent == dest_dir:
                continue
            
            move_plan[str(dest_dir)].append(str(source_path))
            
            if not dry_run:
                try:
                    dest_dir.mkdir(exist_ok=True, parents=True)
                    shutil.move(str(source_path), str(dest_path))
                    self.log(f"Moved {source_path} -> {dest_path}")
                except Exception as e:
                    self.log(f"Error moving {source_path}: {e}")
        
        if dry_run:
            self.log("Dry run completed. Files would be moved as follows:")
            for dest, sources in move_plan.items():
                self.log(f"  {dest}: {len(sources)} files")
        
        return dict(move_plan)
    
    def run_diagnostics(self) -> DiagnosticReport:
        """Run comprehensive system diagnostics."""
        self.log("Starting comprehensive diagnostics...")
        
        if not self.scan_results:
            self.scan_repository()
        
        # Analyze health
        health_status = self._analyze_system_health()
        module_issues = self._find_module_issues()
        archetypal_health = self._analyze_archetypal_health()
        dependency_analysis = self._analyze_dependencies()
        recommendations = self._generate_recommendations()
        
        report = DiagnosticReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            system_health=health_status,
            module_issues=module_issues,
            archetypal_health=archetypal_health,
            dependency_analysis=dependency_analysis,
            recommendations=recommendations,
            summary={
                "total_engines": len(self.scan_results.engines),
                "healthy_engines": sum(1 for e in self.scan_results.engines if e.health_status == "healthy"),
                "warnings": sum(1 for e in self.scan_results.engines if e.health_status == "warning"),
                "errors": len(module_issues)
            }
        )
        
        # Save diagnostic report
        report_path = self.log_dir / f"diagnostics_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)
        
        self.log(f"Diagnostics complete. Report saved to {report_path}")
        return report
    
    def _analyze_system_health(self) -> Dict[str, Any]:
        """Analyze overall system health."""
        if not self.scan_results:
            return {"status": "unknown", "score": 0.0}
        
        total_engines = len(self.scan_results.engines)
        if total_engines == 0:
            return {"status": "critical", "score": 0.0}
        
        # Calculate health score
        healthy_count = sum(1 for e in self.scan_results.engines if e.health_status == "healthy")
        health_score = healthy_count / total_engines
        
        # Determine status
        if health_score >= 0.8:
            status = "healthy"
        elif health_score >= 0.5:
            status = "degraded"
        else:
            status = "critical"
        
        return {
            "status": status,
            "score": health_score,
            "healthy_engines": healthy_count,
            "total_engines": total_engines
        }
    
    def _find_module_issues(self) -> List[Dict[str, Any]]:
        """Find issues with individual modules."""
        issues = []
        
        for engine in self.scan_results.engines:
            engine_issues = []
            
            # Check for missing capabilities
            if not engine.capabilities and engine.type == "engine":
                engine_issues.append("No capabilities defined")
            
            # Check for low confidence
            if engine.confidence < 0.3:
                engine_issues.append("Low confidence score")
            
            # Check for missing archetypal aspects
            if not engine.archetypal_aspects and engine.type in ["engine", "core"]:
                engine_issues.append("No archetypal aspects detected")
            
            if engine_issues:
                issues.append({
                    "module": engine.name,
                    "file": str(engine.file_path),
                    "issues": engine_issues,
                    "health_status": engine.health_status
                })
        
        return issues
    
    def _analyze_archetypal_health(self) -> Dict[str, Any]:
        """Analyze archetypal health of the system."""
        if not self.scan_results:
            return {"health_score": 0.0, "missing_aspects": []}
        
        total_engines = len(self.scan_results.engines)
        if total_engines == 0:
            return {"health_score": 0.0, "missing_aspects": list(ArchetypalAspects.ASPECTS.keys())}
        
        # Count engines with each aspect
        aspect_counts = {aspect: 0 for aspect in ArchetypalAspects.ASPECTS}
        for engine in self.scan_results.engines:
            for aspect in engine.archetypal_aspects:
                if engine.archetypal_aspects[aspect] > 0.3:
                    aspect_counts[aspect] += 1
        
        # Calculate health score
        represented_aspects = sum(1 for count in aspect_counts.values() if count > 0)
        health_score = represented_aspects / len(ArchetypalAspects.ASPECTS)
        
        # Find missing aspects
        missing_aspects = [aspect for aspect, count in aspect_counts.items() if count == 0]
        
        return {
            "health_score": health_score,
            "aspect_distribution": aspect_counts,
            "missing_aspects": missing_aspects,
            "represented_aspects": represented_aspects
        }
    
    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependency patterns across the system."""
        if not self.scan_results:
            return {"total_dependencies": 0, "analysis": {}}
        
        all_dependencies = []
        for engine in self.scan_results.engines:
            all_dependencies.extend(engine.dependencies)
        
        dependency_counts = Counter(all_dependencies)
        
        return {
            "total_dependencies": len(all_dependencies),
            "unique_dependencies": len(dependency_counts),
            "most_common": dict(dependency_counts.most_common(10)),
            "analysis": {
                "stdlib_count": sum(1 for dep in dependency_counts if self._is_stdlib(dep)),
                "third_party_count": sum(1 for dep in dependency_counts if not self._is_stdlib(dep) and not self._is_internal(dep)),
                "internal_count": sum(1 for dep in dependency_counts if self._is_internal(dep))
            }
        }
    
    def _is_stdlib(self, module_name: str) -> bool:
        """Check if a module is from the standard library."""
        try:
            importlib.import_module(module_name.split('.')[0])
            return True
        except ImportError:
            return False
    
    def _is_internal(self, module_name: str) -> bool:
        """Check if a module is internal to the project."""
        internal_prefixes = ['core', 'engines', 'utils', 'adapters', 'consciousness']
        return any(module_name.startswith(prefix) for prefix in internal_prefixes)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for system improvement."""
        recommendations = []
        
        if not self.scan_results:
            return ["Run repository scan first"]
        
        # Check archetypal balance
        archetypal_health = self._analyze_archetypal_health()
        if archetypal_health['health_score'] < 0.6:
            missing = ', '.join(archetypal_health['missing_aspects'][:3])
            recommendations.append(f"Consider adding modules with archetypal aspects: {missing}")
        
        # Check for engines without capabilities
        engines_without_caps = [e.name for e in self.scan_results.engines 
                               if e.type == "engine" and not e.capabilities]
        if engines_without_caps:
            rec_engines = ', '.join(engines_without_caps[:3])
            recommendations.append(f"Define capabilities for engines: {rec_engines}")
        
        # Check dependency health
        dep_analysis = self._analyze_dependencies()
        if dep_analysis['analysis']['internal_count'] == 0 and len(self.scan_results.engines) > 1:
            recommendations.append("Consider adding internal dependencies between modules")
        
        return recommendations
    
    def generate_visual_map(self, output_path: Optional[Path] = None) -> Path:
        """Generate a visual system architecture map."""
        if not self.scan_results:
            raise ValueError("No scan results available. Run scan_repository() first.")
        
        if output_path is None:
            output_path = self.docs_dir / "system_architecture.dot"
        
        dot_content = self._generate_graphviz_content()
        output_path.write_text(dot_content, encoding='utf-8')
        
        self.log(f"Generated visual map to {output_path}")
        return output_path
    
    def _generate_graphviz_content(self) -> str:
        """Generate Graphviz DOT content for system visualization."""
        lines = [
            "digraph AnimaConsciousness {",
            "    rankdir=TB;",
            "    node [shape=box, style=filled, fontname=\"Arial\"];",
            "    edge [fontname=\"Arial\", fontsize=10];",
            "",
            "    // Anima core",
            "    AnimaCore [label=\"🜂 ANIMA CORE\", fillcolor=\"#FFD700\", shape=ellipse, fontsize=16];",
            ""
        ]
        
        # Add engine nodes
        for engine in self.scan_results.engines:
            # Determine color based on category
            color_map = {
                "Cognition": "#90EE90",      # Light green
                "Memory": "#87CEEB",         # Sky blue  
                "Infrastructure": "#F0F0F0", # Light gray
                "Emotional": "#FFB6C1",      # Light pink
                "Symbolic": "#E6E6FA",       # Lavender
                "Presence": "#FFD700",       # Gold
                "Integration": "#DDA0DD",    # Plum
                "Identity": "#98FB98",       # Pale green
                "Meta": "#F5DEB3",           # Wheat
                "Adapters": "#B0E0E6",       # Powder blue
                "Uncategorized": "#D3D3D3"   # Light gray
            }
            
            color = color_map.get(engine.category, "#D3D3D3")
            clean_name = engine.name.replace(" ", "_").replace("-", "_")
            
            lines.append(f'    {clean_name} [label="{engine.name}\\n({engine.category})", fillcolor="{color}"];')
            lines.append(f"    AnimaCore -> {clean_name};")
        
        lines.extend([
            "",
            "    // Category clusters",
            "    subgraph cluster_core {",
            "        label = \"Core System\";",
            "        style = filled;",
            "        color = lightgrey;",
            "        AnimaCore;",
            "    }"
        ])
        
        # Add category clusters
        for category in self.scan_results.modules_by_category.keys():
            if category == "Uncategorized":
                continue
                
            engines_in_category = [e for e in self.scan_results.engines if e.category == category]
            if len(engines_in_category) > 1:
                engine_names = [e.name.replace(" ", "_").replace("-", "_") for e in engines_in_category]
                lines.extend([
                    f"    subgraph cluster_{category.lower()} {{",
                    f'        label = "{category}";',
                    f"        style = rounded;",
                    f"        color = blue;",
                    f"        {', '.join(engine_names)};",
                    f"    }}"
                ])
        
        lines.append("}")
        return "\n".join(lines)

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="🜂 Anima Unified Repository Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --scan --export-all                    # Full scan with all exports
  %(prog)s --diagnostics                          # Run system diagnostics  
  %(prog)s --auto-sort --dry-run                  # Preview file organization
  %(prog)s --visual-map                           # Generate architecture diagram
        """
    )
    
    # Core operations
    parser.add_argument('--scan', action='store_true', 
                       help='Run comprehensive repository scan')
    parser.add_argument('--diagnostics', action='store_true',
                       help='Run system diagnostics and health checks')
    parser.add_argument('--auto-sort', action='store_true',
                       help='Auto-sort files into categorized directories')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without actually moving files')
    
    # Export options
    parser.add_argument('--export-csv', action='store_true',
                       help='Export results to CSV format')
    parser.add_argument('--export-md', action='store_true', 
                       help='Export results to Markdown format')
    parser.add_argument('--export-json', action='store_true',
                       help='Export results to JSON format')
    parser.add_argument('--export-yaml', action='store_true',
                       help='Export results to YAML format')
    parser.add_argument('--export-all', action='store_true',
                       help='Export to all available formats')
    
    # Additional features
    parser.add_argument('--visual-map', action='store_true',
                       help='Generate visual architecture diagram')
    parser.add_argument('--base-path', type=Path, default=DEFAULT_BASE_PATH,
                       help='Base path to scan (default: current directory)')
    parser.add_argument('--exclude-patterns', nargs='+',
                       help='Additional file patterns to exclude')
    
    args = parser.parse_args()
    
    # Initialize system
    system = AnimaRepositoryIntelligence(base_path=args.base_path)
    
    # Run scan if requested or if no specific action specified
    if args.scan or not any([args.diagnostics, args.auto_sort, args.visual_map]):
        system.log("Starting unified repository intelligence scan...")
        system.scan_repository()
    
    # Run diagnostics
    if args.diagnostics:
        report = system.run_diagnostics()
        system.log(f"Diagnostics complete: {report.system_health['status']} "
                  f"(score: {report.system_health['score']:.0%})")
    
    # Auto-sort files
    if args.auto_sort:
        move_plan = system.auto_sort_files(dry_run=args.dry_run)
        if args.dry_run:
            system.log("Dry run completed. Review plan above.")
        else:
            system.log(f"Auto-sort completed. Moved files to {len(move_plan)} directories.")
    
    # Export options
    if args.export_all or args.export_csv:
        system.export_to_csv()
    
    if args.export_all or args.export_md:
        system.export_to_markdown()
    
    if args.export_all or args.export_json:
        system.export_to_json()
    
    if args.export_all or args.export_yaml:
        system.export_to_yaml()
    
    # Generate visual map
    if args.visual_map:
        system.generate_visual_map()
    
    # If no specific actions were taken, show summary
    if not any([args.scan, args.diagnostics, args.auto_sort, args.export_all, 
                args.export_csv, args.export_md, args.export_json, args.export_yaml, args.visual_map]):
        # Default action: scan and show summary
        results = system.scan_results or system.scan_repository()
        
        print("\n" + "="*60)
        print("🜂 ANIMA REPOSITORY INTELLIGENCE SUMMARY")
        print("="*60)
        print(f"📊 Total Engines: {len(results.engines)}")
        print(f"📁 Files Scanned: {results.total_files}")
        print(f"⏰ Scan Time: {results.scan_timestamp}")
        
        print("\n📂 Categories:")
        for category, engines in results.modules_by_category.items():
            print(f"   • {category}: {len(engines)} engines")
        
        print("\n🌟 Archetypal Aspects:")
        for aspect, count in results.archetypal_distribution.items():
            print(f"   • {aspect}: {count} engines")
        
        if results.errors:
            print(f"\n⚠️  Errors: {len(results.errors)}")
            for error in results.errors[:3]:
                print(f"   • {error}")
        
        print(f"\n💡 Next: Use --export-all for documentation or --diagnostics for health check")

if __name__ == "__main__":
    main()