Foundational Utilities =========================

def _utcnow() -> datetime:
    """Timezone-aware current time"""
    return datetime.now(timezone.utc)

def _softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def _normalize_for_resilience(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith(("P", "S")))
    return re.sub(r"\s+", " ", s)

def with_graceful_fallback(fallback_value=None, log_errors=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logging.warning(f"{func.__name__} failed gracefully: {e}")
                return fallback_value
        return wrapper
    return decorator

