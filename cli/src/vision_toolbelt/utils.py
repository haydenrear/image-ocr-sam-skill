from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps

SCHEMA_VERSION = "vision-toolbelt.analysis.v1"


def pathify(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_image(path: str | Path) -> Image.Image:
    img = Image.open(path)
    return ImageOps.exif_transpose(img).convert("RGB")


def image_metadata(path: str | Path) -> dict[str, Any]:
    p = pathify(path)
    img = load_image(p)
    mime, _ = mimetypes.guess_type(str(p))
    return {
        "path": str(p),
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "mime": mime or "application/octet-stream",
        "sha256": sha256_file(p),
    }


def ensure_parent(path: str | Path) -> Path:
    p = pathify(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Any) -> Path:
    p = ensure_parent(path)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bbox_norm(box: Iterable[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [round(x1 / width, 6), round(y1 / height, 6), round(x2 / width, 6), round(y2 / height, 6)]


def clamp_box(box: Iterable[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, min(width, x1)); x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1)); y2 = max(0, min(height, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def unavailable(engine: str, reason: str) -> dict[str, Any]:
    return {"engine": engine, "status": "unavailable", "reason": reason}


def default_model_cache() -> Path:
    env = os.environ.get("VISION_TOOLBELT_MODEL_CACHE")
    if env:
        return pathify(env)
    return Path.home() / ".cache" / "vision-toolbelt" / "models"


def analysis_base(image_path: str | Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "image": image_metadata(image_path),
        "caption": {"engine": "none", "status": "not_requested", "text": ""},
        "ocr": [],
        "regions": [],
        "artifacts": {},
        "warnings": [],
    }
