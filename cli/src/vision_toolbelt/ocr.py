from __future__ import annotations

from typing import Any
from pathlib import Path

from .utils import load_image, command_exists, clamp_box


def ocr_tesseract(image_path: str | Path, lang: str = "eng", min_conf: float = 0.0) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if not command_exists("tesseract"):
        return [], ["tesseract binary is not on PATH"]
    try:
        import pytesseract
        from pytesseract import Output
    except Exception as e:
        return [], [f"pytesseract is not installed: {e}"]

    img = load_image(image_path)
    try:
        data = pytesseract.image_to_data(img, lang=lang, output_type=Output.DICT)
    except Exception as e:
        return [], [f"tesseract OCR failed: {e}"]

    out: list[dict[str, Any]] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data.get("conf", [0])[i]) / 100.0
        except Exception:
            conf = 0.0
        if conf < min_conf:
            continue
        x = int(data["left"][i]); y = int(data["top"][i])
        w = int(data["width"][i]); h = int(data["height"][i])
        box = clamp_box([x, y, x + w, y + h], img.width, img.height)
        out.append({
            "id": f"ocr_{len(out)+1:03d}",
            "engine": "tesseract",
            "text": text,
            "bbox_xyxy": box,
            "confidence": round(conf, 4),
            "block_num": data.get("block_num", [None] * n)[i],
            "line_num": data.get("line_num", [None] * n)[i],
            "word_num": data.get("word_num", [None] * n)[i],
        })
    return out, warnings


def ocr_paddle(image_path: str | Path, lang: str = "en") -> tuple[list[dict[str, Any]], list[str]]:
    try:
        from paddleocr import PaddleOCR
    except Exception as e:
        return [], [f"paddleocr is not installed: {e}"]
    img = load_image(image_path)
    try:
        engine = PaddleOCR(use_angle_cls=True, lang=lang)
        result = engine.ocr(str(image_path), cls=True)
    except Exception as e:
        return [], [f"paddleocr failed: {e}"]
    blocks: list[dict[str, Any]] = []
    for page in result or []:
        for item in page or []:
            quad, payload = item
            text, conf = payload
            xs = [p[0] for p in quad]; ys = [p[1] for p in quad]
            box = clamp_box([min(xs), min(ys), max(xs), max(ys)], img.width, img.height)
            blocks.append({
                "id": f"ocr_{len(blocks)+1:03d}",
                "engine": "paddleocr",
                "text": text,
                "bbox_xyxy": box,
                "quad": quad,
                "confidence": round(float(conf), 4),
            })
    return blocks, []


def run_ocr(image_path: str | Path, engine: str = "auto", lang: str = "eng", min_conf: float = 0.0) -> tuple[list[dict[str, Any]], list[str], str]:
    engine = (engine or "auto").lower()
    warnings: list[str] = []
    if engine in {"none", "off", "false"}:
        return [], [], "none"
    engines = ["tesseract", "paddleocr"] if engine == "auto" else [engine]
    for e in engines:
        if e == "tesseract":
            blocks, w = ocr_tesseract(image_path, lang=lang, min_conf=min_conf)
        elif e == "paddleocr":
            paddle_lang = "en" if lang == "eng" else lang
            blocks, w = ocr_paddle(image_path, lang=paddle_lang)
        else:
            blocks, w = [], [f"unknown OCR engine: {e}"]
        if blocks:
            return blocks, warnings + w, e
        warnings.extend(w)
    return [], warnings, engines[0] if engines else "none"
