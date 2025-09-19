
# =========================
# SecretVault
# =========================

class SecretVault:
    def __init__(self, vault_key: Optional[str] = None):
        self.vault_key = (vault_key or os.getenv("ANIMA_VAULT_KEY") or "dev-vault-key").encode("utf-8")
        self._store: Dict[str, Dict[str, str]] = {}

    def _derive(self, canon_value: str, salt: bytes) -> str:
        dk = hashlib.pbkdf2_hmac("sha256", canon_value.encode("utf-8"), self.vault_key + salt, 200_000)
        return dk.hex()

    def register_secret(self, label: str, value: str) -> None:
        salt = os.urandom(16)
        digest = self._derive(_canon(value), salt)
        self._store[label] = {"salt": salt.hex(), "digest": digest}

    def verify(self, label: str, candidate: str) -> bool:
        rec = self._store.get(label)
        if not rec:
            return False
        salt = bytes.fromhex(rec["salt"])
        digest = self._derive(_canon(candidate), salt)
        return _ct_eq(digest, rec["digest"])

    def has(self, label: str) -> bool:
        return label in self._store
