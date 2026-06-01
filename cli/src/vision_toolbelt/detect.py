from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import load_image, bbox_norm, clamp_box, default_model_cache


def _regions_from_boxes(boxes: list[list[float]], labels: list[str], scores: list[float] | None, width: int, height: int, source: str) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for i, box in enumerate(boxes):
        xyxy = clamp_box(box, width, height)
        regions.append({
            "id": f"r{i+1:03d}",
            "label": labels[i] if i < len(labels) else "region",
            "bbox_xyxy": xyxy,
            "bbox_norm": bbox_norm(xyxy, width, height),
            "confidence": round(float(scores[i]), 4) if scores and i < len(scores) else None,
            "source": source,
            "description": labels[i] if i < len(labels) else "Detected region",
        })
    return regions


def florence_detect(image_path: str | Path, prompt: str = "", model_id: str = "microsoft/Florence-2-base", device: str = "auto", task: str | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
    except Exception as e:
        return [], [f"florence2 requires transformers/torch: {e}"]
    img = load_image(image_path)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    if task is None:
        task = "<CAPTION_TO_PHRASE_GROUNDING>" if prompt.strip() else "<OD>"
    text = task + (prompt.strip() if task == "<CAPTION_TO_PHRASE_GROUNDING>" else "")
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir=str(default_model_cache()))
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True, cache_dir=str(default_model_cache())).to(device)
        inputs = processor(text=text, images=img, return_tensors="pt").to(device, dtype)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=1024, do_sample=False, num_beams=3,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(generated_text, task=task, image_size=(img.width, img.height))
        payload = parsed.get(task, parsed) if isinstance(parsed, dict) else {}
        boxes = payload.get("bboxes", []) if isinstance(payload, dict) else []
        labels = payload.get("labels", []) if isinstance(payload, dict) else []
        return _regions_from_boxes(boxes, labels, None, img.width, img.height, "florence2"), []
    except Exception as e:
        return [], [f"florence2 detection failed: {e}"]


def grounding_dino_detect(image_path: str | Path, prompt: str, model_id: str = "IDEA-Research/grounding-dino-tiny", box_threshold: float = 0.25, text_threshold: float = 0.25, device: str = "auto") -> tuple[list[dict[str, Any]], list[str]]:
    if not prompt.strip():
        return [], ["grounding-dino requires --prompt with target phrases"]
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except Exception as e:
        return [], [f"grounding-dino requires transformers/torch: {e}"]
    img = load_image(image_path)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(default_model_cache()))
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, cache_dir=str(default_model_cache())).to(device)
        text = prompt.strip()
        if not text.endswith("."):
            text += "."
        inputs = processor(images=img, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[img.height, img.width]], device=device)
        try:
            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=box_threshold, text_threshold=text_threshold, target_sizes=target_sizes
            )[0]
        except TypeError:
            results = processor.post_process_grounded_object_detection(
                outputs, threshold=box_threshold, text_threshold=text_threshold, target_sizes=target_sizes
            )[0]
        boxes = results.get("boxes", []).detach().cpu().tolist()
        labels = results.get("labels", [])
        scores = results.get("scores", []).detach().cpu().tolist()
        return _regions_from_boxes(boxes, labels, scores, img.width, img.height, "grounding-dino"), []
    except Exception as e:
        return [], [f"grounding-dino detection failed: {e}"]


def yolo_detect(image_path: str | Path, model_id: str = "yolov8n.pt", conf: float = 0.25) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        from ultralytics import YOLO
    except Exception as e:
        return [], [f"ultralytics is not installed: {e}"]
    img = load_image(image_path)
    try:
        model = YOLO(model_id)
        results = model(str(image_path), conf=conf, verbose=False)
        regions: list[dict[str, Any]] = []
        for res in results:
            names = getattr(res, "names", {})
            for b in res.boxes:
                xyxy = b.xyxy[0].detach().cpu().tolist()
                cls = int(b.cls[0].detach().cpu().item()) if getattr(b, "cls", None) is not None else -1
                label = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
                score = float(b.conf[0].detach().cpu().item()) if getattr(b, "conf", None) is not None else None
                regions.extend(_regions_from_boxes([xyxy], [label], [score] if score is not None else None, img.width, img.height, "yolo"))
        for i, r in enumerate(regions):
            r["id"] = f"r{i+1:03d}"
        return regions, []
    except Exception as e:
        return [], [f"yolo detection failed: {e}"]


def run_detect(image_path: str | Path, engine: str = "auto", prompt: str = "", model_id: str | None = None, conf: float = 0.25, device: str = "auto") -> tuple[list[dict[str, Any]], list[str], str]:
    engine = (engine or "auto").lower()
    if engine in {"none", "off", "false"}:
        return [], [], "none"
    warnings: list[str] = []
    engines = ["florence2", "grounding-dino", "yolo"] if engine == "auto" else [engine]
    for e in engines:
        if e in {"florence", "florence2"}:
            regions, w = florence_detect(image_path, prompt=prompt, model_id=model_id or "microsoft/Florence-2-base", device=device)
        elif e in {"grounding-dino", "groundingdino", "dino"}:
            regions, w = grounding_dino_detect(image_path, prompt=prompt, model_id=model_id or "IDEA-Research/grounding-dino-tiny", box_threshold=conf, text_threshold=conf, device=device)
            e = "grounding-dino"
        elif e == "yolo":
            regions, w = yolo_detect(image_path, model_id=model_id or "yolov8n.pt", conf=conf)
        else:
            regions, w = [], [f"unknown detection engine: {e}"]
        if regions:
            return regions, warnings + w, e
        warnings.extend(w)
    return [], warnings, engines[0] if engines else "none"
