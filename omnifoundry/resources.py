"""Versioned resources: prompt templates and the task catalog. Both are data, hashed into lineage.

Prompts live in `prompts/<name>.md` and use str.format placeholders ({name}; literal braces doubled).
`prompts.overrides` in the config maps a name to another file, so an experiment can swap one prompt
without touching the rest.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import get, resolve_path
from .jsonio import digest


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str
    path: str

    @property
    def hash(self):
        return digest("prompt", self.text)[:16]

    def render(self, **values):
        return self.text.format(**values)


class Prompts:
    def __init__(self, config):
        self.dir = resolve_path(get(config, "prompts.dir", "prompts"))
        self.overrides = get(config, "prompts.overrides", {}) or {}
        self._cache = {}

    def __getitem__(self, name):
        if name not in self._cache:
            path = resolve_path(self.overrides[name]) if name in self.overrides else self.dir / f"{name}.md"
            text = Path(path).read_text(encoding="utf-8")
            if text.startswith("<!--"):           # strip a leading provenance comment
                text = text.split("-->", 1)[1].lstrip("\n")
            self._cache[name] = Prompt(name, text, str(path))
        return self._cache[name]

    def hashes(self, *names):
        return {n: self[n].hash for n in names}


class Catalog:
    """Task types. Each: id, route, name, kind (anchor|holistic), definition, rules, capabilities, enabled."""

    def __init__(self, config):
        path = resolve_path(get(config, "catalog.path", "catalog/tasks.yaml"))
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.tasks = {t["id"]: t for t in data["tasks"]}
        self.hash = digest("catalog", data)[:16]
        enabled = get(config, "catalog.enabled")
        if enabled:
            unknown = set(enabled) - set(self.tasks)
            if unknown:
                raise ValueError(f"catalog.enabled names unknown tasks: {sorted(unknown)}")
            self.tasks = {k: v for k, v in self.tasks.items() if k in enabled}

    def for_route(self, route):
        return [t for t in self.tasks.values() if t.get("route", "rov") == route and t.get("enabled", True)]

    def describe(self, route):
        """Catalog text block for prompts."""
        lines = []
        for t in self.for_route(route):
            lines.append(f"- id: {t['id']}  ({t.get('kind', 'anchor')})\n  name: {t['name']}\n  definition: {t['definition'].strip()}")
            for rule in t.get("rules", []):
                lines.append(f"  rule: {rule}")
        return "\n".join(lines)
