---
name: vision-toolbelt
description: Use when an agent needs OCR, screenshots, image captioning/understanding, PDF-to-Markdown extraction, bounding boxes, segmentation masks, or standardized image/document analysis artifacts across Claude Code, Codex, Gemini, or text-only models.
---

# Vision Toolbelt Skill

Use this skill whenever you need to inspect an image, screenshot, UI, photo, repair image, support-thread image, document image, PDF, or any task requiring OCR, image understanding, PDF-to-Markdown extraction, bounding boxes, regions, crops, overlays, or masks.

## Core rule

Do not invent precise image coordinates from raw vision alone. Use the `vision-toolbelt` CLI to produce evidence artifacts, then reason over the JSON and overlays.

## Standard workflow

1. Normalize the image and collect metadata:

```bash
vision-toolbelt inspect path/to/image.png --out artifacts/image.inspect.json
```

2. Run a standardized analysis package:

```bash
vision-toolbelt analyze path/to/image.png \
  --prompt "battery, connector, screw, cracked plastic, corrosion, missing part" \
  --ocr-engine auto \
  --caption-engine auto \
  --detect-engine auto \
  --segment-engine none \
  --overlay-out artifacts/image.overlay.png \
  --out artifacts/image.analysis.json
```

3. For bounding boxes, use `detect` or `analyze`; for masks, run `segment` after boxes exist:

```bash
vision-toolbelt detect path/to/image.png \
  --prompt "swollen battery, loose ribbon cable, cracked hinge" \
  --engine auto \
  --out artifacts/image.boxes.json

vision-toolbelt segment path/to/image.png \
  --boxes artifacts/image.boxes.json \
  --engine auto \
  --out-dir artifacts/masks \
  --out artifacts/image.masks.json
```

4. Render an overlay before trusting boxes:

```bash
vision-toolbelt overlay path/to/image.png \
  --analysis artifacts/image.analysis.json \
  --out artifacts/image.overlay.png
```

5. For non-multimodal models, pass the JSON summary, OCR blocks, region descriptions, bounding boxes, and overlay path instead of the original image.

6. For PDFs, extract Markdown before asking a model to reason over the document:

```bash
vision-toolbelt pdf-markdown report.pdf \
  --out artifacts/report.md \
  --meta-out artifacts/report.document.json
```

## CLI discovery

The CLI is self-documenting. Use:

```bash
vision-toolbelt --help
vision-toolbelt analyze --help
vision-toolbelt toolspec --format json
vision-toolbelt schema --format json
vision-toolbelt models list --format table
vision-toolbelt pdf-markdown --help
```

Every command that creates an artifact accepts `--out` or `--out-dir`. Prefer explicit output paths so Claude Code, Codex, Gemini, and text-only models all consume the same artifact layout.

## Required schema conventions

Bounding boxes are always `bbox_xyxy = [x1, y1, x2, y2]` in source-image pixels unless explicitly marked as normalized. Normalized boxes are always `[x1 / width, y1 / height, x2 / width, y2 / height]` and stored as `bbox_norm`.

Each analysis JSON should contain:

```json
{
  "schema_version": "vision-toolbelt.analysis.v1",
  "image": {"path": "...", "width": 1280, "height": 720, "sha256": "..."},
  "caption": {"engine": "...", "text": "...", "status": "ok|unavailable|error"},
  "ocr": [{"id": "ocr_001", "text": "...", "bbox_xyxy": [0, 0, 10, 10], "confidence": 0.9}],
  "regions": [{"id": "r001", "label": "...", "bbox_xyxy": [0, 0, 10, 10], "bbox_norm": [0, 0, 0.1, 0.1]}],
  "artifacts": {"overlay_path": "..."},
  "warnings": []
}
```

## Engine selection

Use `auto` unless the user requests a specific engine. `auto` chooses the cheapest available local option and reports unavailable engines instead of silently hallucinating.

Recommended low-cost bundle:

- OCR: Tesseract when available; Florence-2 OCR as an optional model-backed fallback.
- Image understanding: Florence-2 base for local captioning, OCR, object detection, and phrase grounding.
- Segmentation: SAM 2.1 Hiera Tiny for box-prompted masks, or rectangular mask fallback when SAM is unavailable.
- PDF extraction: PyMuPDF4LLM for local PDF-to-Markdown conversion, with optional image extraction and OCR language hints for scanned pages.

## Agent-specific guidance

Claude Code, Codex, and Gemini may already understand images. Still use this CLI for coordinates, repeatable OCR, crops, overlays, and machine-readable artifacts. For text-only models such as DeepSeek, run `vision-toolbelt analyze` first and pass the analysis JSON plus overlay/crop paths as the visual context.
