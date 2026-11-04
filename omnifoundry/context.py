"""Run context: everything a stage or check needs, built once from the config."""
from dataclasses import dataclass, field
from pathlib import Path

from .cache import Cache
from .config import get, load_env_file, resolve_path
from .jsonio import write_json
from .lineage import code_version
from .media import Media
from .policies.router import ModelRouter
from .resources import Catalog, Prompts


@dataclass
class Context:
    config: dict
    run_dir: Path
    cache: Cache
    media: Media
    prompts: Prompts
    catalog: Catalog
    models: ModelRouter
    work_dir: Path            # derived media (transcodes, clips); shared across runs, content-addressed
    state: dict = field(default_factory=dict)

    @classmethod
    def build(cls, config, run_dir):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        load_env_file(resolve_path(get(config, "env_file", ".env")))
        cache = Cache(resolve_path(get(config, "cache.dir", "cache")), enabled=get(config, "cache.enabled", True))
        ctx = cls(
            config=config,
            run_dir=run_dir,
            cache=cache,
            media=Media(get(config, "media", {})),
            prompts=Prompts(config),
            catalog=Catalog(config),
            models=ModelRouter(get(config, "models", {}), cache, ledger=run_dir / "calls.jsonl"),
            work_dir=resolve_path(get(config, "media.work_dir", "cache/media")),
        )
        return ctx

    def get(self, dotted, default=None):
        return get(self.config, dotted, default)

    def snapshot(self, command):
        write_json(self.run_dir / "run.json", {
            "command": command,
            "config": self.config,
            "code_version": code_version(),
            "catalog_hash": self.catalog.hash,
        })
