"""Settings, read from environment variables with local-friendly defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    # Any OpenAI-compatible endpoint. Defaults to a local FreeLLMAPI gateway.
    base_url: str = field(default_factory=lambda: _env("DOCPLAT_BASE_URL", "http://localhost:3001/v1"))
    api_key: str = field(default_factory=lambda: _env("DOCPLAT_API_KEY", ""))
    chat_model: str = field(default_factory=lambda: _env("DOCPLAT_CHAT_MODEL", "auto:smart"))
    fast_model: str = field(default_factory=lambda: _env("DOCPLAT_FAST_MODEL", "auto:fast"))
    # Empty = use the built-in local embedding, so the app works even if the
    # gateway has no embedding model configured.
    embed_model: str = field(default_factory=lambda: _env("DOCPLAT_EMBED_MODEL", ""))
    data_dir: Path = field(default_factory=lambda: Path(_env("DOCPLAT_DATA_DIR", "data_store")))
    max_agent_steps: int = field(default_factory=lambda: int(_env("DOCPLAT_MAX_AGENT_STEPS", "30")))
    keep_raw_files: bool = field(default_factory=lambda: _env("DOCPLAT_KEEP_RAW", "0") == "1")
    request_timeout: float = 180.0
