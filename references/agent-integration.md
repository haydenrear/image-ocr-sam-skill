# Agent integration notes

## Universal policy

Agents should not hand-estimate exact coordinates. They should run `vision-toolbelt`, inspect the overlay, and cite `region.id`, `bbox_xyxy`, and `bbox_norm` in downstream answers.

## Multimodal agents

Claude Code, Codex, and Gemini can use native image understanding for broad semantic reasoning. Use this toolbelt for repeatability: OCR, bounding boxes, segmentation masks, overlays, crops, and artifact handoff to other models.

## Text-only agents

For DeepSeek or other text-only models, convert images into a compact text + JSON bundle:

```bash
vision-toolbelt analyze image.png \
  --prompt "objects relevant to the user's question" \
  --out image.analysis.json \
  --overlay-out image.overlay.png
```

Then pass:

- `image.caption.text`
- OCR block text and coordinates
- region labels, descriptions, and boxes
- overlay/crop paths
- warnings/unavailable engines

## Cost strategy

Run in tiers:

1. `inspect` and `ocr` first.
2. `caption` only when a semantic description is needed.
3. `detect` only when boxes are needed.
4. `segment` only after boxes exist or the user explicitly needs masks.

## Re-run strategy

If boxes look poor, crop the suspicious region and rerun detection on the crop:

```bash
vision-toolbelt crop image.png --bbox 420,180,730,410 --out crops/r001.png
vision-toolbelt analyze crops/r001.png --prompt "specific defect" --out crops/r001.analysis.json
```
