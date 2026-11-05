"""Model calls: one interface for every role, OpenAI-compatible transport, mock transport, request cache.

A request is a list of parts:
    {"type": "text", "text": ...} | {"type": "image"|"audio"|"video", "path": ...}
Each backend turns parts into its wire format. The cache key covers the model options (minus
timeouts), the parts (files by content hash), the JSON-schema hint and `sample` (draw index), so
repeated draws at temperature > 0 are distinct cache entries and reruns replay them exactly.
"""
import base64
import importlib
import mimetypes
import os
from pathlib import Path
import threading
import time

from .jsonio import append_jsonl, digest, file_hash, parse_json_text

NON_IDENTITY = {"timeout_s", "max_retries", "concurrency", "qps", "api_key_env", "base_url"}


class LLMError(RuntimeError):
    def __init__(self, message, retryable=False):
        super().__init__(message)
        self.retryable = retryable


class Result(dict):
    """{'text', 'json' (or None), 'model', 'usage', 'cached', 'elapsed_s'}"""


def _part_identity(part):
    if part["type"] == "text":
        return part
    return {"type": part["type"], "sha256": file_hash(part["path"])}


def _data_url(path, default_mime):
    mime = mimetypes.guess_type(str(path))[0] or default_mime
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


class _RateLimiter:
    def __init__(self, concurrency, qps):
        self.sem = threading.BoundedSemaphore(max(1, int(concurrency)))
        self.interval = 1.0 / qps if qps else 0.0
        self.lock = threading.Lock()
        self.next_at = 0.0

    def __enter__(self):
        self.sem.acquire()
        if self.interval:
            with self.lock:
                now = time.monotonic()
                wait = self.next_at - now
                self.next_at = max(now, self.next_at) + self.interval
            if wait > 0:
                time.sleep(wait)

    def __exit__(self, *exc):
        self.sem.release()


_limiters = {}
_limiters_lock = threading.Lock()


def _limiter(options):
    key = (options.get("base_url"), options.get("model"))
    with _limiters_lock:
        if key not in _limiters:
            _limiters[key] = _RateLimiter(options.get("concurrency", 8), options.get("qps", 0))
        return _limiters[key]


class OpenAICompatible:
    """Chat-completions transport. Media wire formats are configurable because gateways differ:
    video_format: file | image_url | video_url ; audio_format: input_audio | file.
    Verify with scripts/probe_gateway.py: a gateway may accept a format yet silently drop the media."""

    def __init__(self, options):
        from openai import OpenAI
        key = os.environ.get(options.get("api_key_env", "OPENAI_API_KEY"))
        if not key:
            raise LLMError(f"missing credential env var {options.get('api_key_env')}")
        self.options = options
        self.client = OpenAI(api_key=key, base_url=options.get("base_url"), timeout=options.get("timeout_s", 300),
                             max_retries=0)

    def _content(self, parts):
        o = self.options
        out = []
        for p in parts:
            t = p["type"]
            if t == "text":
                out.append({"type": "text", "text": p["text"]})
            elif t == "image":
                out.append({"type": "image_url", "image_url": {"url": _data_url(p["path"], "image/jpeg")}})
            elif t == "audio":
                path = Path(p["path"])
                if o.get("audio_format", "input_audio") == "file":
                    out.append({"type": "file", "file": {"filename": path.name, "file_data": _data_url(path, "audio/mpeg")}})
                else:
                    fmt = path.suffix.lstrip(".").lower() or "mp3"
                    data = base64.b64encode(path.read_bytes()).decode("ascii")
                    out.append({"type": "input_audio", "input_audio": {"data": data, "format": fmt}})
            elif t == "video":
                url = _data_url(p["path"], "video/mp4")
                fmt = o.get("video_format", "file")
                if fmt == "file":
                    out.append({"type": "file", "file": {"filename": Path(p["path"]).name, "file_data": url}})
                elif fmt == "image_url":
                    out.append({"type": "image_url", "image_url": {"url": url}})
                else:
                    out.append({"type": "video_url", "video_url": {"url": url}})
            else:
                raise ValueError(f"unknown part type {t}")
        return out

    def complete(self, parts, *, temperature=None, json_mode=False):
        o = self.options
        params = {k: o[k] for k in ("max_tokens", "top_p", "reasoning_effort") if k in o}
        temp = o.get("temperature") if temperature is None else temperature
        if temp is not None:
            params["temperature"] = temp
        if o.get("extra_body"):
            params["extra_body"] = o["extra_body"]
        try:
            r = self.client.chat.completions.create(model=o["model"], messages=[{"role": "user", "content": self._content(parts)}],
                                                    **params)
        except Exception as exc:  # noqa: BLE001 - classify, never leak bodies/keys
            code = getattr(exc, "status_code", None)
            retryable = code is None or code == 429 or (isinstance(code, int) and code >= 500)
            raise LLMError(f"{o['model']} request failed (status={code}): {type(exc).__name__}", retryable) from exc
        if not r.choices:
            raise LLMError(f"{o['model']} returned no choices", retryable=True)
        choice = r.choices[0]
        text = choice.message.content or ""
        if choice.finish_reason not in ("stop", None):
            raise LLMError(f"{o['model']} finish_reason={choice.finish_reason}", retryable=choice.finish_reason == "length")
        usage = r.usage.model_dump(mode="json") if r.usage else {}
        return text, {"model": r.model or o["model"], "usage": usage}


class MockBackend:
    """Calls `module:function(role, parts, options) -> str`. Used by tests and dry runs."""

    def __init__(self, options):
        mod, fn = options["responder"].split(":")
        self.fn = getattr(importlib.import_module(mod), fn)
        self.options = options

    def complete(self, parts, *, temperature=None, json_mode=False):
        return self.fn(self.options.get("role"), parts, self.options), {"model": "mock:" + self.options.get("model", ""), "usage": {}}


BACKENDS = {"openai": OpenAICompatible, "mock": MockBackend}


class Client:
    """One role's model endpoint with caching, retries and a call ledger."""

    def __init__(self, role, options, cache, ledger=None):
        self.role = role
        self.options = {**options, "role": role}
        self.cache = cache
        self.ledger = ledger
        self._backend = None
        self.identity = {k: v for k, v in self.options.items() if k not in NON_IDENTITY and k != "role"}

    @property
    def model(self):
        return self.options.get("model", "")

    def backend(self):
        if self._backend is None:
            self._backend = BACKENDS[self.options.get("backend", "openai")](self.options)
        return self._backend

    def key(self, parts, *, sample=0, temperature=None, tag=""):
        return digest("llm/v1", self.identity, [_part_identity(p) for p in parts], sample, temperature, tag)

    def call(self, parts, *, json=True, sample=0, temperature=None, tag="", context=None):
        """Returns Result. Raises LLMError after retries. `tag` names the purpose (e.g. check name)."""
        key = self.key(parts, sample=sample, temperature=temperature, tag=tag)
        hit = self.cache.get("llm", key)
        if hit is not None:
            res = Result(hit["value"], cached=True)
            self._log(key, res, tag, context, None)
            return res
        retries = int(self.options.get("max_retries", 3))
        last = None
        for attempt in range(retries + 1):
            t0 = time.monotonic()
            try:
                with _limiter(self.options):
                    text, meta = self.backend().complete(parts, temperature=temperature, json_mode=json)
                parsed = None
                if json:
                    try:
                        parsed = parse_json_text(text)
                    except ValueError as exc:
                        raise LLMError(f"{self.model} returned invalid JSON: {exc}", retryable=True) from exc
                res = Result(text=text, json=parsed, model=meta["model"], usage=meta.get("usage", {}),
                             elapsed_s=round(time.monotonic() - t0, 3))
                self.cache.put("llm", key, {"value": dict(res), "request": {"role": self.role, "tag": tag}})
                res["cached"] = False
                self._log(key, res, tag, context, None)
                return res
            except LLMError as exc:
                last = exc
                self._log(key, None, tag, context, f"{exc} (attempt {attempt + 1})")
                if not exc.retryable or attempt == retries:
                    break
                # Gateway hiccups (503/429, hung connections) last tens of seconds: back off 5, 10, 20, 40, 80 s.
                time.sleep(min(120, 5 * 2 ** attempt))
        raise last

    def _log(self, key, res, tag, context, error):
        if self.ledger is None:
            return
        row = {"t": round(time.time(), 3), "role": self.role, "model": self.model, "tag": tag, "key": key,
               "context": context or {}}
        if res is not None:
            row.update(cached=res.get("cached", False), usage=res.get("usage", {}), elapsed_s=res.get("elapsed_s"))
        if error:
            row["error"] = error
        append_jsonl(self.ledger, row)
