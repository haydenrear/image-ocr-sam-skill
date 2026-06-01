from __future__ import annotations

TOOL_SPEC = {
    "schema_version": "vision-toolbelt.toolspec.v1",
    "commands": [
        {
            "name": "inspect",
            "purpose": "Read image metadata and write standard image object.",
            "required_args": ["image"],
            "important_options": ["--out"],
            "output": "JSON image metadata with path, width, height, mime, sha256.",
        },
        {
            "name": "ocr",
            "purpose": "Run OCR and return text blocks with pixel coordinates.",
            "required_args": ["image"],
            "important_options": ["--engine auto|tesseract|paddleocr", "--lang", "--out"],
            "output": "JSON list under ocr[].",
        },
        {
            "name": "caption",
            "purpose": "Produce semantic image description for multimodal or text-only handoff.",
            "required_args": ["image"],
            "important_options": ["--engine auto|basic|florence2", "--detail", "--out"],
            "output": "JSON caption object.",
        },
        {
            "name": "detect",
            "purpose": "Produce bounding boxes. Never estimate coordinates manually when this command is available.",
            "required_args": ["image"],
            "important_options": ["--prompt", "--engine auto|florence2|grounding-dino|yolo", "--out"],
            "output": "Analysis-style JSON with regions[].bbox_xyxy and regions[].bbox_norm.",
        },
        {
            "name": "segment",
            "purpose": "Produce masks from existing boxes. Use after detect or with a user-supplied boxes JSON.",
            "required_args": ["image", "--boxes"],
            "important_options": ["--engine auto|sam2_hf|box", "--out-dir", "--out"],
            "output": "Analysis-style JSON with regions[].mask_path.",
        },
        {
            "name": "analyze",
            "purpose": "Run inspect + OCR + caption + detection + optional segmentation + overlay in one standardized package.",
            "required_args": ["image"],
            "important_options": ["--prompt", "--out", "--overlay-out", "--ocr-engine", "--caption-engine", "--detect-engine", "--segment-engine"],
            "output": "Full vision-toolbelt.analysis.v1 JSON.",
        },
        {
            "name": "overlay",
            "purpose": "Render boxes and OCR blocks onto an image for validation.",
            "required_args": ["image", "--analysis", "--out"],
            "output": "Annotated PNG/JPEG path.",
        },
        {
            "name": "crop",
            "purpose": "Crop a bbox region for focused re-analysis.",
            "required_args": ["image", "--bbox", "--out"],
            "output": "Cropped image path plus metadata JSON when --json-out is used.",
        },
        {
            "name": "screenshot",
            "purpose": "Capture current screen using native OS command and optionally analyze it.",
            "required_args": ["--out"],
            "important_options": ["--analyze", "--analysis-out"],
            "output": "Screenshot path and optional analysis JSON.",
        },
    ],
    "agent_rules": [
        "Do not invent precise coordinates; run detect/analyze.",
        "Render overlay before trusting boxes.",
        "Use bbox_xyxy for pixels and bbox_norm for normalized coordinates.",
        "For text-only models, pass caption, OCR, regions, and overlay/crop paths.",
        "Use segment only after boxes exist unless the user explicitly needs full-image masks.",
    ],
}
