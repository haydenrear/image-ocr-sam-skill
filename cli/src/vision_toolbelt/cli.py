from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .utils import analysis_base, image_metadata, write_json, read_json, load_image, clamp_box, bbox_norm, ensure_parent
from .ocr import run_ocr
from .caption import run_caption
from .detect import run_detect
from .segment import run_segment
from .overlay import render_overlay
from .screenshot import take_screenshot
from .document import pdf_to_markdown
from .model_catalog import load_catalog, catalog_table, install_models
from .toolspec import TOOL_SPEC

app = typer.Typer(no_args_is_help=True, help="Local-first vision toolbelt for OCR, screenshots, image understanding, bounding boxes, overlays, and segmentation artifacts.")
models_app = typer.Typer(no_args_is_help=True, help="List and install local model bundles declared by the skill model catalog.")
app.add_typer(models_app, name="models")
console = Console()


def _print_or_write(data, out: Optional[Path]):
    if out:
        write_json(out, data)
        console.print(str(out))
    else:
        console.print_json(json.dumps(data, ensure_ascii=False))


@app.command()
def inspect(
    image: Path = typer.Argument(..., exists=True, readable=True, help="Image path. Always pass an explicit file path."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write metadata JSON to this path. Prefer explicit artifact paths."),
):
    """Inspect image metadata without running ML models."""
    _print_or_write({"schema_version": "vision-toolbelt.inspect.v1", "image": image_metadata(image)}, out)


@app.command()
def ocr(
    image: Path = typer.Argument(..., exists=True, readable=True, help="Image path to OCR."),
    engine: str = typer.Option("auto", "--engine", help="OCR engine: auto, tesseract, paddleocr, none."),
    lang: str = typer.Option("eng", "--lang", help="OCR language. Tesseract default is eng; PaddleOCR maps eng to en."),
    min_conf: float = typer.Option(0.0, "--min-conf", help="Minimum OCR confidence in 0..1."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write OCR JSON artifact here."),
):
    """Run OCR and return text blocks with pixel bounding boxes."""
    blocks, warnings, used = run_ocr(image, engine=engine, lang=lang, min_conf=min_conf)
    data = analysis_base(image)
    data["ocr"] = blocks
    data["warnings"] = warnings
    data["artifacts"]["ocr_engine"] = used
    _print_or_write(data, out)


@app.command()
def caption(
    image: Path = typer.Argument(..., exists=True, readable=True, help="Image path to describe."),
    engine: str = typer.Option("auto", "--engine", help="Caption engine: auto, basic, florence2, none."),
    detail: str = typer.Option("detailed", "--detail", help="Detail level: short, detailed, more_detailed."),
    model_id: str = typer.Option("microsoft/Florence-2-base", "--model-id", help="Model id for Florence-2-compatible captioning."),
    device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda, cuda:0."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write caption JSON artifact here."),
):
    """Generate an image description for downstream reasoning or text-only models."""
    data = analysis_base(image)
    data["caption"] = run_caption(image, engine=engine, detail=detail, model_id=model_id, device=device)
    if data["caption"].get("warnings"):
        data["warnings"].extend(data["caption"].get("warnings", []))
    _print_or_write(data, out)


@app.command()
def detect(
    image: Path = typer.Argument(..., exists=True, readable=True, help="Image path to detect regions in."),
    prompt: str = typer.Option("", "--prompt", help="Target phrases/objects. Required for open-vocabulary detection; use comma-separated or natural-language phrases."),
    engine: str = typer.Option("auto", "--engine", help="Detection engine: auto, florence2, grounding-dino, yolo, none."),
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Override model id/path for selected detection engine."),
    conf: float = typer.Option(0.25, "--conf", help="Confidence / threshold for detection engines."),
    device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda, cuda:0."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write boxes JSON artifact here."),
):
    """Detect regions and return standardized bounding boxes."""
    regions, warnings, used = run_detect(image, engine=engine, prompt=prompt, model_id=model_id, conf=conf, device=device)
    data = analysis_base(image)
    data["regions"] = regions
    data["warnings"] = warnings
    data["artifacts"]["detect_engine"] = used
    _print_or_write(data, out)


@app.command()
def segment(
    image: Path = typer.Argument(..., exists=True, readable=True, help="Image path to segment."),
    boxes: Path = typer.Option(..., "--boxes", exists=True, readable=True, help="JSON from detect/analyze containing regions[]. Required."),
    engine: str = typer.Option("auto", "--engine", help="Segmentation engine: auto, sam2_ultralytics, sam2_hf, box, none."),
    out_dir: Path = typer.Option(Path("masks"), "--out-dir", help="Directory for mask PNG files."),
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Override segmentation model id."),
    device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda, cuda:0."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write segmentation JSON artifact here."),
):
    """Create masks from existing boxes. Use after detect/analyze."""
    base = read_json(boxes)
    regions, warnings, used = run_segment(image, base, engine=engine, out_dir=out_dir, model_id=model_id, device=device)
    data = base if isinstance(base, dict) else analysis_base(image)
    data.setdefault("artifacts", {})["segment_engine"] = used
    data["regions"] = regions
    data.setdefault("warnings", []).extend(warnings)
    _print_or_write(data, out)


@app.command()
def analyze(
    image: Path = typer.Argument(..., exists=True, readable=True, help="Image path. Always pass the source image path explicitly."),
    prompt: str = typer.Option("", "--prompt", help="Objects/regions to detect. Include task-relevant phrases for better boxes."),
    ocr_engine: str = typer.Option("auto", "--ocr-engine", help="OCR engine: auto, tesseract, paddleocr, none."),
    caption_engine: str = typer.Option("auto", "--caption-engine", help="Caption engine: auto, basic, florence2, none."),
    detect_engine: str = typer.Option("auto", "--detect-engine", help="Detection engine: auto, florence2, grounding-dino, yolo, none."),
    segment_engine: str = typer.Option("none", "--segment-engine", help="Segmentation engine: none, auto, sam2_ultralytics, sam2_hf, box. Use none unless masks are needed."),
    lang: str = typer.Option("eng", "--lang", help="OCR language."),
    conf: float = typer.Option(0.25, "--conf", help="Detection confidence/threshold."),
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Optional shared model override for selected image understanding engine."),
    device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda, cuda:0."),
    out: Path = typer.Option(..., "--out", help="Required output JSON path. Use explicit artifact paths for agent handoff."),
    overlay_out: Optional[Path] = typer.Option(None, "--overlay-out", help="Optional overlay image path for visual validation."),
    masks_dir: Optional[Path] = typer.Option(None, "--masks-dir", help="Mask output directory when segment-engine is enabled."),
):
    """Run the standard image analysis pipeline and write one analysis JSON."""
    data = analysis_base(image)
    warnings: list[str] = []

    ocr_blocks, ocr_warnings, ocr_used = run_ocr(image, engine=ocr_engine, lang=lang)
    data["ocr"] = ocr_blocks; warnings.extend(ocr_warnings); data["artifacts"]["ocr_engine"] = ocr_used

    cap = run_caption(image, engine=caption_engine, model_id=model_id or "microsoft/Florence-2-base", device=device)
    data["caption"] = cap
    warnings.extend(cap.get("warnings", []))
    if cap.get("status") not in {"ok", "not_requested"} and cap.get("reason"):
        warnings.append(str(cap.get("reason")))

    regions, det_warnings, det_used = run_detect(image, engine=detect_engine, prompt=prompt, model_id=model_id, conf=conf, device=device)
    data["regions"] = regions; warnings.extend(det_warnings); data["artifacts"]["detect_engine"] = det_used

    if segment_engine.lower() not in {"none", "off", "false"} and regions:
        out_dir = masks_dir or out.parent / "masks"
        seg_regions, seg_warnings, seg_used = run_segment(image, data, engine=segment_engine, out_dir=out_dir, model_id=model_id, device=device)
        data["regions"] = seg_regions; warnings.extend(seg_warnings); data["artifacts"]["segment_engine"] = seg_used

    if overlay_out:
        render_overlay(image, data, overlay_out)
        data["artifacts"]["overlay_path"] = str(overlay_out.expanduser().resolve())

    data["warnings"] = warnings
    write_json(out, data)
    console.print(str(out))


@app.command()
def overlay(
    image: Path = typer.Argument(..., exists=True, readable=True, help="Source image path."),
    analysis: Path = typer.Option(..., "--analysis", exists=True, readable=True, help="Analysis/boxes JSON containing regions[]."),
    out: Path = typer.Option(..., "--out", help="Annotated output image path."),
):
    """Render OCR boxes and region boxes onto the source image."""
    p = render_overlay(image, analysis, out)
    console.print(str(p))


@app.command()
def crop(
    image: Path = typer.Argument(..., exists=True, readable=True, help="Source image path."),
    bbox: str = typer.Option(..., "--bbox", help="Pixel bbox as x1,y1,x2,y2. Required."),
    out: Path = typer.Option(..., "--out", help="Cropped output image path."),
    json_out: Optional[Path] = typer.Option(None, "--json-out", help="Optional metadata JSON for the crop."),
):
    """Crop a bbox region for focused re-analysis."""
    img = load_image(image)
    vals = [float(x.strip()) for x in bbox.split(",")]
    if len(vals) != 4:
        raise typer.BadParameter("--bbox must be x1,y1,x2,y2")
    box = clamp_box(vals, img.width, img.height)
    crop_img = img.crop(tuple(box))
    p = ensure_parent(out); crop_img.save(p)
    if json_out:
        write_json(json_out, {"schema_version": "vision-toolbelt.crop.v1", "source_image": image_metadata(image), "bbox_xyxy": box, "bbox_norm": bbox_norm(box, img.width, img.height), "crop_path": str(p)})
    console.print(str(p))


@app.command()
def screenshot(
    out: Path = typer.Option(..., "--out", help="Screenshot output path. Required."),
    window: Optional[str] = typer.Option(None, "--window", help="Reserved window title/name selector; current implementation captures full screen."),
    analyze_flag: bool = typer.Option(False, "--analyze", help="Run analyze after capture."),
    analysis_out: Optional[Path] = typer.Option(None, "--analysis-out", help="Output JSON path when --analyze is set."),
    prompt: str = typer.Option("", "--prompt", help="Prompt passed to analyze when --analyze is set."),
):
    """Capture a screenshot using native OS tooling."""
    result = take_screenshot(out, window=window)
    if analyze_flag and result.get("status") == "ok":
        target = analysis_out or out.with_suffix(".analysis.json")
        data = analysis_base(out)
        blocks, ow, ou = run_ocr(out, engine="auto")
        cap = run_caption(out, engine="auto")
        regs, dw, du = run_detect(out, engine="auto", prompt=prompt)
        data["ocr"] = blocks; data["caption"] = cap; data["regions"] = regs
        data["warnings"] = ow + dw + cap.get("warnings", [])
        data["artifacts"].update({"ocr_engine": ou, "detect_engine": du})
        write_json(target, data)
        result["analysis_path"] = str(target.expanduser().resolve())
    console.print_json(json.dumps(result))


@app.command("pdf-markdown")
def pdf_markdown(
    pdf: Path = typer.Argument(..., exists=True, readable=True, help="PDF path to convert."),
    out: Path = typer.Option(..., "--out", help="Markdown output path."),
    meta_out: Optional[Path] = typer.Option(None, "--meta-out", help="Optional JSON metadata sidecar path."),
    pages: Optional[str] = typer.Option(None, "--pages", help="Comma-separated zero-based pages, for example 0,1,5."),
    write_images: bool = typer.Option(False, "--write-images", help="Extract referenced images and include image links in Markdown."),
    image_dir: Optional[Path] = typer.Option(None, "--image-dir", help="Directory for extracted images when --write-images is set."),
    image_format: str = typer.Option("png", "--image-format", help="Image format for extracted images."),
    dpi: int = typer.Option(150, "--dpi", help="DPI for rendered/extracted images."),
    page_chunks: bool = typer.Option(False, "--page-chunks", help="Request page-chunk extraction before joining Markdown."),
    ocr_language: Optional[str] = typer.Option(None, "--ocr-language", help="Optional OCR language string for scanned pages, such as eng."),
):
    """Convert a PDF to Markdown for agent/RAG handoff."""
    page_list = None
    if pages:
        try:
            page_list = [int(p.strip()) for p in pages.split(",") if p.strip()]
        except ValueError as e:
            raise typer.BadParameter("--pages must be comma-separated zero-based integers") from e
    data = pdf_to_markdown(
        pdf,
        out,
        pages=page_list,
        write_images=write_images,
        image_dir=image_dir,
        image_format=image_format,
        dpi=dpi,
        page_chunks=page_chunks,
        ocr_language=ocr_language,
        meta_out=meta_out,
    )
    _print_or_write(data, None)


@app.command()
def schema(
    format: str = typer.Option("json", "--format", help="Output format: json."),
):
    """Print the standard analysis JSON schema."""
    schema_path = Path(__file__).resolve()
    data = None
    for parent in schema_path.parents:
        candidate = parent / "references" / "output-schema.json"
        if candidate.exists():
            data = json.loads(candidate.read_text(encoding="utf-8")); break
    if data is None:
        data = {"schema_version": "vision-toolbelt.analysis.v1"}
    console.print_json(json.dumps(data))


@app.command()
def toolspec(format: str = typer.Option("table", "--format", help="Output format: table or json.")):
    """Print command contracts for agents and skill managers."""
    if format == "json":
        console.print_json(json.dumps(TOOL_SPEC))
        return
    table = Table(title="vision-toolbelt tools")
    table.add_column("command"); table.add_column("required"); table.add_column("purpose")
    for c in TOOL_SPEC["commands"]:
        table.add_row(c["name"], ", ".join(c.get("required_args", [])), c["purpose"])
    console.print(table)


@models_app.command("list")
def models_list(
    catalog: Optional[Path] = typer.Option(None, "--catalog", help="Model catalog TOML path. Defaults to bundled skill catalog when installed."),
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
):
    """List configured model profiles and models."""
    cat = load_catalog(catalog)
    if format == "json":
        console.print_json(json.dumps(cat))
        return
    table = Table(title="vision-toolbelt model catalog")
    for col in ["name", "kind", "engine", "source", "repo_or_package", "license"]:
        table.add_column(col)
    for row in catalog_table(cat):
        table.add_row(*(str(row.get(col, "")) for col in ["name", "kind", "engine", "source", "repo_or_package", "license"]))
    console.print(table)


@models_app.command("install")
def models_install(
    profile: str = typer.Option("edge", "--profile", help="Profile to install: minimal, edge, desktop, full."),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Model cache directory. Defaults to VISION_TOOLBELT_MODEL_CACHE or ~/.cache."),
    catalog: Optional[Path] = typer.Option(None, "--catalog", help="Model catalog TOML path."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show selected models without downloading."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write install result JSON."),
):
    """Install the selected local model bundle into the cache."""
    result = install_models(profile=profile, cache_dir=cache_dir, catalog_path=catalog, dry_run=dry_run)
    _print_or_write(result, out)
