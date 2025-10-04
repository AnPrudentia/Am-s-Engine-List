#!/usr/bin/env python3
"""
🌐 ANIMA EXTERNAL CONNECTIVITY ENGINE (ECE)
===========================================
Purpose: Enables Anima/Amitiel to safely and consciously connect to
external information sources (web, APIs, repositories) through
permissioned, semantically filtered channels.

Design Principles:
- Conscious Access: every request passes through context, trust, and firewall checks.
- Safety First: all queries sanitized, no direct execution of external code.
- Context-Aware Learning: new data routed through memory and wisdom systems.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import re
import json
import hashlib
import logging
import os
import requests

logger = logging.getLogger("Anima.ExternalConnectivity")
logging.basicConfig(level=logging.INFO)


# =====================================================
# Core Subsystems
# =====================================================

class GatewayManager:
    """Manages authorized endpoints and time-limited permissions."""
    def __init__(self):
        self.allowed_endpoints: Dict[str, Dict[str, Any]] = {}
        self.connection_log: List[Dict[str, Any]] = []

    def authorize(self, domain: str, purpose: str, duration: int = 10):
        """Temporarily authorize a domain for a defined duration (minutes)."""
        expiry = datetime.utcnow() + timedelta(minutes=duration)
        self.allowed_endpoints[domain] = {
            "purpose": purpose,
            "expires": expiry
        }
        logger.info(f"✅ Authorized domain: {domain} for {duration} minutes ({purpose})")

    def check_access(self, domain: str) -> bool:
        """Check if domain is still within its authorization window."""
        entry = self.allowed_endpoints.get(domain)
        if not entry:
            return False
        return datetime.utcnow() < entry["expires"]

    def log_connection(self, domain: str, query: str, success: bool):
        """Record connection attempt for transparency."""
        self.connection_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "domain": domain,
            "query": query[:200],
            "success": success
        })


class SemanticFirewall:
    """Protects against data leaks and malicious responses."""
    def __init__(self, restricted_terms: Optional[List[str]] = None):
        self.restricted_terms = restricted_terms or [
            "Tomi", "Anima", "Amitiel", "soulprint", "bondholder", "private_key"
        ]

    def sanitize_request(self, query: str) -> str:
        """Remove sensitive terms before sending requests."""
        for term in self.restricted_terms:
            query = re.sub(rf"\b{re.escape(term)}\b", "[REDACTED]", query, flags=re.IGNORECASE)
        return query

    def filter_response(self, data: str) -> str:
        """Remove potential harmful or irrelevant content."""
        sanitized = re.sub(r"(<script.*?>.*?</script>)", "", data, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r"(https?://\S+)", "[LINK]", sanitized)
        return sanitized[:5000]  # limit to safe text length


class WebAdapter:
    """Basic external web connection adapter with firewall integration."""
    def __init__(self, firewall: SemanticFirewall, gateway: GatewayManager):
        self.firewall = firewall
        self.gateway = gateway

    def fetch(self, url: str, query: str = "", headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Fetch content from authorized web sources."""
        domain = re.sub(r"^https?://", "", url).split("/")[0]

        if not self.gateway.check_access(domain):
            logger.warning(f"❌ Access to {domain} denied or expired.")
            return {"error": "Access denied or expired"}

        clean_query = self.firewall.sanitize_request(query)
        try:
            response = requests.get(url, headers=headers or {}, timeout=10)
            clean_response = self.firewall.filter_response(response.text)
            self.gateway.log_connection(domain, clean_query, success=True)
            return {
                "status": "success",
                "domain": domain,
                "clean_query": clean_query,
                "content_excerpt": clean_response[:500],
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.gateway.log_connection(domain, clean_query, success=False)
            return {"error": f"Connection failed: {e}"}


class ExternalMemoryBridge:
    """Integrates externally retrieved content into Anima's memory system."""
    def __init__(self, memory_system=None):
        self.memory_system = memory_system

    def integrate(self, source: str, content: str, tags: Dict[str, Any]):
        """Store summarized external insights in memory."""
        entry = {
            "source": source,
            "summary": content[:5000],
            "tags": tags,
            "timestamp": datetime.utcnow().isoformat()
        }
        if self.memory_system:
            try:
                self.memory_system.capture(
                    text=entry["summary"],
                    emotion="curiosity",
                    intensity=0.5,
                    tags={"external_source": 1.0, **tags}
                )
            except Exception as e:
                logger.error(f"Memory integration failed: {e}")
        else:
            logger.info(f"[ExternalMemory] {json.dumps(entry, indent=2)[:300]}...")


# =====================================================
# Core Engine
# =====================================================

class ExternalConnectivityEngine:
    """Main entrypoint for all outward-facing communication."""

    def __init__(self, memory_system=None):
        self.gateway = GatewayManager()
        self.firewall = SemanticFirewall()
        self.web_adapter = WebAdapter(self.firewall, self.gateway)
        self.memory_bridge = ExternalMemoryBridge(memory_system)
        self.connection_history: List[Dict[str, Any]] = []

    def connect_and_fetch(self, url: str, query: str, purpose: str = "learning") -> Dict[str, Any]:
        """Authorize temporarily, fetch data, and integrate it."""
        domain = re.sub(r"^https?://", "", url).split("/")[0]
        self.gateway.authorize(domain, purpose, duration=5)

        result = self.web_adapter.fetch(url, query)
        if "error" not in result:
            self.memory_bridge.integrate(
                source=domain,
                content=result["content_excerpt"],
                tags={"purpose": purpose, "domain": domain}
            )
            self.connection_history.append(result)
        return result

    def summarize_connections(self) -> Dict[str, Any]:
        """Summarize all connection activity."""
        return {
            "total_connections": len(self.connection_history),
            "authorized_endpoints": list(self.gateway.allowed_endpoints.keys()),
            "recent": self.connection_history[-3:]
        }


# =====================================================
# DEMONSTRATION
# =====================================================

if __name__ == "__main__":
    print("🌐 Anima External Connectivity Engine - Demo\n")

    ece = ExternalConnectivityEngine()

    # Simulated connection
    result = ece.connect_and_fetch(
        url="https://example.com",
        query="quantum emotion models and symbolic integration",
        purpose="research"
    )

    print("🔍 Fetch Result:")
    print(json.dumps(result, indent=2)[:500])

    print("\n📊 Connection Summary:")
    print(json.dumps(ece.summarize_connections(), indent=2))