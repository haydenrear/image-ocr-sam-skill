from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import load_image, default_model_cache


def basic_caption(image_path: str | Path) -> dict[str, Any]:
    img = load_image(image_path)
    small = img.resize((1, 1))
    r, g, b = small.getpixel((0, 0))
    orientation = "landscape" if img.width > img.height else "portrait" if img.height > img.width else "square"
    return {
        "engine": "basic",
        "status": "ok",
        "text": f"Image {img.width}x{img.height} ({orientation}). Average RGB approximately ({r}, {g}, {b}). No semantic model was used.",
        "detailed": "Basic fallback caption only reports image metadata and coarse color statistics.",
    }


def florence_caption(image_path: str | Path, model_id: str = "microsoft/Florence-2-base", detail: str = "detailed", device: str = "auto") -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
    except Exception as e:
        return {"engine": "florence2", "status": "unavailable", "text": "", "reason": f"transformers/torch not installed: {e}"}
    img = load_image(image_path)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    prompt = {
        "short": "<CAPTION>",
        "caption": "<CAPTION>",
        "detailed": "<DETAILED_CAPTION>",
        "more": "<MORE_DETAILED_CAPTION>",
        "more_detailed": "<MORE_DETAILED_CAPTION>",
    }.get(detail, "<DETAILED_CAPTION>")
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir=str(default_model_cache()))
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True, cache_dir=str(default_model_cache())).to(device)
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device, dtype)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            do_sample=False,
            num_beams=3,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(generated_text, task=prompt, image_size=(img.width, img.height))
        if isinstance(parsed, dict):
            text = parsed.get(prompt) or next(iter(parsed.values()), "")
        else:
            text = str(parsed)
        return {"engine": "florence2", "model_id": model_id, "status": "ok", "text": str(text), "raw": parsed}
    except Exception as e:
        return {"engine": "florence2", "model_id": model_id, "status": "error", "text": "", "reason": str(e)}


def run_caption(image_path: str | Path, engine: str = "auto", detail: str = "detailed", model_id: str = "microsoft/Florence-2-base", device: str = "auto") -> dict[str, Any]:
    engine = (engine or "auto").lower()
    if engine in {"none", "off", "false"}:
        return {"engine": "none", "status": "not_requested", "text": ""}
    if engine == "basic":
        return basic_caption(image_path)
    if engine in {"auto", "florence", "florence2"}:
        result = florence_caption(image_path, model_id=model_id, detail=detail, device=device)
        if result.get("status") == "ok" or engine in {"florence", "florence2"}:
            return result
        fallback = basic_caption(image_path)
        fallback["warnings"] = [result.get("reason", "florence2 unavailable")]
        return fallback
    return {"engine": engine, "status": "unavailable", "text": "", "reason": f"unknown caption engine: {engine}"}
