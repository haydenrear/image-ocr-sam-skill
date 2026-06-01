from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import ImageDraw, ImageFont

from .utils import load_image, ensure_parent, read_json


def render_overlay(image_path: str | Path, analysis: dict[str, Any] | str | Path, out_path: str | Path) -> Path:
    if not isinstance(analysis, dict):
        analysis = read_json(analysis)
    img = load_image(image_path)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    regions = analysis.get("regions", []) or []
    ocr = analysis.get("ocr", []) or []
    for item in regions:
        box = item.get("bbox_xyxy")
        if not box:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        label = item.get("label") or item.get("id") or "region"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 2, max(0, y1 - 12)), f"{item.get('id','')} {label}", fill="red", font=font)
    for item in ocr:
        box = item.get("bbox_xyxy")
        if not box:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=1)
    p = ensure_parent(out_path)
    img.save(p)
    return p
