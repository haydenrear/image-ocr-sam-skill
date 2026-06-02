from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import ensure_parent, pathify, sha256_file, write_json


def pdf_metadata(path: str | Path) -> dict[str, Any]:
    p = pathify(path)
    meta: dict[str, Any] = {
        "path": str(p),
        "sha256": sha256_file(p),
        "mime": "application/pdf",
    }
    try:
        import pymupdf

        doc = pymupdf.open(p)
        meta["page_count"] = doc.page_count
        if doc.metadata:
            meta["metadata"] = {k: v for k, v in doc.metadata.items() if v}
        doc.close()
    except Exception as e:
        meta["page_count"] = None
        meta["warning"] = f"PDF metadata unavailable: {e}"
    return meta


def pdf_to_markdown(
    pdf_path: str | Path,
    out: str | Path,
    *,
    pages: list[int] | None = None,
    write_images: bool = False,
    image_dir: str | Path | None = None,
    image_format: str = "png",
    dpi: int = 150,
    page_chunks: bool = False,
    ocr_language: str | None = None,
    meta_out: str | Path | None = None,
) -> dict[str, Any]:
    pdf = pathify(pdf_path)
    md_path = ensure_parent(out)
    warnings: list[str] = []
    try:
        import pymupdf4llm
    except Exception as e:
        raise RuntimeError(f"pymupdf4llm is not installed: {e}") from e

    kwargs: dict[str, Any] = {
        "pages": pages,
        "write_images": write_images,
        "image_format": image_format,
        "dpi": dpi,
        "page_chunks": page_chunks,
    }
    if image_dir:
        kwargs["image_path"] = str(Path(image_dir).expanduser().resolve())
    if ocr_language:
        kwargs["ocr_language"] = ocr_language
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    result = pymupdf4llm.to_markdown(str(pdf), **kwargs)
    if page_chunks:
        if isinstance(result, list):
            markdown = "\n\n".join(str(chunk.get("text", "")) for chunk in result if isinstance(chunk, dict))
        else:
            warnings.append("page_chunks requested, but extractor did not return a list")
            markdown = str(result)
    else:
        markdown = str(result)
    md_path.write_text(markdown, encoding="utf-8")

    data = {
        "schema_version": "vision-toolbelt.document.v1",
        "document": pdf_metadata(pdf),
        "engine": "pymupdf4llm",
        "markdown_path": str(md_path),
        "markdown_chars": len(markdown),
        "artifacts": {
            "markdown_path": str(md_path),
            "image_dir": str(Path(image_dir).expanduser().resolve()) if image_dir else None,
        },
        "warnings": warnings,
    }
    if meta_out:
        write_json(meta_out, data)
    return data
