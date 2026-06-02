from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .utils import load_image, read_json, write_json, ensure_parent, default_model_cache


def _regions_from_boxes_json(boxes_json: str | Path | dict[str, Any]) -> list[dict[str, Any]]:
    data = read_json(boxes_json) if not isinstance(boxes_json, dict) else boxes_json
    if "regions" in data:
        return data["regions"] or []
    if "boxes" in data:
        return data["boxes"] or []
    return []


def box_masks(image_path: str | Path, regions: list[dict[str, Any]], out_dir: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    img = load_image(image_path)
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    updated: list[dict[str, Any]] = []
    for i, r in enumerate(regions):
        box = r.get("bbox_xyxy")
        if not box:
            continue
        mask = Image.new("L", (img.width, img.height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([int(v) for v in box], fill=255)
        rid = r.get("id") or f"r{i+1:03d}"
        mask_path = out / f"{rid}.mask.png"
        mask.save(mask_path)
        rr = dict(r)
        rr["id"] = rid
        rr["mask_path"] = str(mask_path)
        rr["mask_engine"] = "box"
        updated.append(rr)
    return updated, []


def sam2_hf_masks(image_path: str | Path, regions: list[dict[str, Any]], out_dir: str | Path, model_id: str = "facebook/sam2.1-hiera-tiny", device: str = "auto") -> tuple[list[dict[str, Any]], list[str]]:
    try:
        import torch
        from transformers import Sam2Model, Sam2Processor
    except Exception as e:
        return [], [f"SAM2 HF requires transformers/torch with Sam2Model support: {e}"]
    img = load_image(image_path)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        processor = Sam2Processor.from_pretrained(model_id, cache_dir=str(default_model_cache()))
        model = Sam2Model.from_pretrained(model_id, cache_dir=str(default_model_cache())).to(device)
    except Exception as e:
        return [], [f"failed to load SAM2 model {model_id}: {e}"]
    out = Path(out_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    updated: list[dict[str, Any]] = []
    for i, r in enumerate(regions):
        box = r.get("bbox_xyxy")
        if not box:
            continue
        rid = r.get("id") or f"r{i+1:03d}"
        try:
            inputs = processor(img, input_boxes=[[[float(v) for v in box]]], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
            mask_tensor = masks[0][0][0]
            mask_img = Image.fromarray((mask_tensor.numpy() > 0.0).astype("uint8") * 255, mode="L")
            mask_path = out / f"{rid}.mask.png"
            mask_img.save(mask_path)
            rr = dict(r); rr["id"] = rid; rr["mask_path"] = str(mask_path); rr["mask_engine"] = "sam2_hf"
            updated.append(rr)
        except Exception as e:
            updated_box, _ = box_masks(image_path, [r], out_dir)
            if updated_box:
                updated_box[0]["mask_warning"] = f"SAM2 failed for {rid}; used rectangular fallback: {e}"
                updated.append(updated_box[0])
    return updated, []


def sam2_ultralytics_masks(image_path: str | Path, regions: list[dict[str, Any]], out_dir: str | Path, model_id: str = "sam2_t.pt", device: str = "auto") -> tuple[list[dict[str, Any]], list[str]]:
    try:
        from ultralytics import SAM
    except Exception as e:
        return [], [f"Ultralytics SAM2 requires ultralytics: {e}"]
    img_path = Path(image_path).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_cache = default_model_cache()
    model_cache.mkdir(parents=True, exist_ok=True)
    updated: list[dict[str, Any]] = []
    try:
        import os

        old_cwd = os.getcwd()
        os.chdir(model_cache)
        try:
            model = SAM(model_id)
        finally:
            os.chdir(old_cwd)
    except Exception as e:
        return [], [f"failed to load Ultralytics SAM2 model {model_id}: {e}"]
    for i, r in enumerate(regions):
        box = r.get("bbox_xyxy")
        if not box:
            continue
        rid = r.get("id") or f"r{i+1:03d}"
        try:
            results = model(str(img_path), bboxes=[[float(v) for v in box]], device=None if device == "auto" else device, verbose=False)
            mask_data = None
            for result in results or []:
                if getattr(result, "masks", None) is not None:
                    mask_data = result.masks.data
                    break
            if mask_data is None or len(mask_data) == 0:
                raise RuntimeError("Ultralytics SAM2 returned no masks")
            arr = (mask_data[0].detach().cpu().numpy() > 0.0).astype("uint8") * 255
            mask_img = Image.fromarray(arr, mode="L")
            mask_path = out / f"{rid}.mask.png"
            mask_img.save(mask_path)
            rr = dict(r)
            rr["id"] = rid
            rr["mask_path"] = str(mask_path)
            rr["mask_engine"] = "sam2_ultralytics"
            updated.append(rr)
        except Exception as e:
            updated_box, _ = box_masks(image_path, [r], out_dir)
            if updated_box:
                updated_box[0]["mask_warning"] = f"Ultralytics SAM2 failed for {rid}; used rectangular fallback: {e}"
                updated.append(updated_box[0])
    return updated, []


def run_segment(image_path: str | Path, boxes_json: str | Path | dict[str, Any], engine: str = "auto", out_dir: str | Path = "masks", model_id: str | None = None, device: str = "auto") -> tuple[list[dict[str, Any]], list[str], str]:
    regions = _regions_from_boxes_json(boxes_json)
    engine = (engine or "auto").lower()
    if engine in {"none", "off", "false"}:
        return regions, [], "none"
    if engine == "box":
        updated, warnings = box_masks(image_path, regions, out_dir)
        return updated, warnings, "box"
    engines = ["sam2_ultralytics", "sam2_hf", "box"] if engine == "auto" else [engine]
    warnings: list[str] = []
    for e in engines:
        if e in {"sam2", "sam2_ultralytics", "sam2-ultralytics", "ultralytics"}:
            updated, w = sam2_ultralytics_masks(image_path, regions, out_dir, model_id=model_id or "sam2_t.pt", device=device)
            e = "sam2_ultralytics"
        elif e in {"sam2_hf", "sam-hf"}:
            updated, w = sam2_hf_masks(image_path, regions, out_dir, model_id=model_id or "facebook/sam2.1-hiera-tiny", device=device)
            e = "sam2_hf"
        elif e == "box":
            updated, w = box_masks(image_path, regions, out_dir)
        else:
            updated, w = [], [f"unknown segmentation engine: {e}"]
        if updated:
            return updated, warnings + w, e
        warnings.extend(w)
    return regions, warnings, engines[0] if engines else "none"
