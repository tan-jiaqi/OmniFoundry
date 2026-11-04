"""YAML configs with `extends` inheritance, deep merge, and per-section hashes.

A config is plain data. Stages hash only the sections they read (`section_hash`), so changing one
stage's settings invalidates that stage and its downstream, never upstream artifacts.
"""
import copy
import os
from pathlib import Path

import yaml

from .jsonio import digest

ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path, overrides=None):
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parents = data.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    merged = {}
    for parent in parents:
        merged = deep_merge(merged, load_config((path.parent / parent).resolve()))
    merged = deep_merge(merged, data)
    for dotted, value in (overrides or {}).items():
        node = merged
        *head, last = dotted.split(".")
        for key in head:
            node = node.setdefault(key, {})
        node[last] = value
    return merged


def get(config, dotted, default=None):
    node = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def section_hash(config, *dotted):
    return digest(*[get(config, d) for d in dotted])


def resolve_path(value):
    """Config paths are relative to the project root unless absolute."""
    p = Path(os.path.expandvars(str(value)))
    return p if p.is_absolute() else ROOT / p


def load_env_file(path):
    """KEY=VALUE or `export KEY=VALUE` lines into os.environ (existing variables win). Values are never logged."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
