"""Content-addressed JSON cache shared by all runs.

Two namespaces are used: `llm` (one entry per model request) and `stage` (one entry per stage item).
Keys are content hashes, so a hit is valid by construction; nothing is ever invalidated in place.
"""
import json
from pathlib import Path

from .jsonio import write_json


class Cache:
    def __init__(self, root, enabled=True):
        self.root = Path(root)
        self.enabled = enabled

    def path(self, ns, key):
        return self.root / ns / key[:2] / f"{key}.json"

    def get(self, ns, key):
        if not self.enabled:
            return None
        p = self.path(ns, key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None   # torn or corrupt entry: recompute

    def put(self, ns, key, value):
        if self.enabled:
            write_json(self.path(ns, key), value)
        return value

    def memo(self, ns, key, fn):
        """(value, hit). Computes and stores on miss."""
        hit = self.get(ns, key)
        if hit is not None:
            return hit["value"], True
        value = fn()
        self.put(ns, key, {"value": value})
        return value, False
