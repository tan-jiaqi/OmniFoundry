"""Provenance attached to every record: what produced it, from which inputs, under which settings."""
from functools import lru_cache
from pathlib import Path

from .jsonio import digest, file_hash

PACKAGE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def code_version():
    """Hash of the package sources. Recorded for audit; deliberately NOT part of cache keys,
    so refactors don't throw away paid model calls. Bump a stage's VERSION to force recompute."""
    files = sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)
    return digest([[str(p.relative_to(PACKAGE)), file_hash(p)] for p in files])[:16]


def make(stage, stage_version, parents, *, prompts=None, models=None, config_hash=None, **extra):
    return {
        "stage": stage,
        "stage_version": stage_version,
        "parents": list(parents),
        "prompts": dict(prompts or {}),     # name -> content hash
        "models": dict(models or {}),       # role -> model id
        "config_hash": config_hash,
        "code_version": code_version(),
        **extra,
    }
