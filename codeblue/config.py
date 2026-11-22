"""TOML experiment config -> model clients and settings."""

from __future__ import annotations

import tomllib
from pathlib import Path

from codeblue.llm import MockCloudModel, MockLocalModel, OpenAICompatModel


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    cfg["_dir"] = str(path.parent.resolve())
    return cfg


def make_model(section: dict, tasks):
    kind = section.get("kind", "mock")
    if kind == "openai":
        return OpenAICompatModel(
            base_url=section["base_url"],
            model=section["model"],
            api_key_env=section.get("api_key_env") or None,
            temperature=section.get("temperature", 0.2),
            max_tokens=section.get("max_tokens", 512),
            timeout=section.get("timeout", 180),
            extra_body=section.get("extra_body"),
        )
    if kind == "mock-local":
        return MockLocalModel(tasks)
    if kind == "mock-cloud":
        return MockCloudModel(tasks)
    raise ValueError(f"unknown model kind {kind!r} (use openai, mock-local or mock-cloud)")
