from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from .utils import default_model_cache


def default_catalog_path() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "references" / "model-catalog.toml"
        if candidate.exists():
            return candidate
    env = os.environ.get("VISION_TOOLBELT_MODEL_CATALOG")
    return Path(env).expanduser().resolve() if env else None


def load_catalog(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path).expanduser().resolve() if path else default_catalog_path()
    if p is None or not p.exists():
        return {"profiles": {}, "models": {}}
    return tomllib.loads(p.read_text(encoding="utf-8"))


def models_for_profile(catalog: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    prof = catalog.get("profiles", {}).get(profile)
    if not prof:
        raise ValueError(f"unknown profile '{profile}'")
    model_names = prof.get("models", []) or []
    models = catalog.get("models", {})
    return [{"name": name, **models[name]} for name in model_names if name in models]


def install_models(profile: str, cache_dir: str | Path | None = None, catalog_path: str | Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    catalog = load_catalog(catalog_path)
    cache = Path(cache_dir).expanduser().resolve() if cache_dir else default_model_cache()
    cache.mkdir(parents=True, exist_ok=True)
    selected = models_for_profile(catalog, profile)
    results: list[dict[str, Any]] = []
    for model in selected:
        source = model.get("source")
        if dry_run:
            results.append({"name": model["name"], "source": source, "status": "dry_run", "target": str(cache)})
            continue
        if source == "hf":
            try:
                from huggingface_hub import snapshot_download
                local = snapshot_download(repo_id=model["repo_id"], cache_dir=str(cache), local_files_only=False)
                results.append({"name": model["name"], "repo_id": model["repo_id"], "status": "ok", "path": local})
            except Exception as e:
                results.append({"name": model["name"], "repo_id": model.get("repo_id"), "status": "error", "reason": str(e)})
        elif source in {"system", "python", "ultralytics"}:
            results.append({"name": model["name"], "source": source, "status": "external", "reason": "no snapshot download required; engine installs or resolves this at runtime"})
        else:
            results.append({"name": model["name"], "source": source, "status": "skipped", "reason": "unknown model source"})
    return {"profile": profile, "cache_dir": str(cache), "results": results}


def catalog_table(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, model in (catalog.get("models", {}) or {}).items():
        rows.append({
            "name": name,
            "kind": model.get("kind", ""),
            "engine": model.get("engine", ""),
            "source": model.get("source", ""),
            "repo_or_package": model.get("repo_id") or model.get("package") or model.get("binary") or model.get("model_id", ""),
            "license": model.get("license", ""),
        })
    return rows
