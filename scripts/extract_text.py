from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.schemas import ExtractionMethod, OCRResult, PageResult  # noqa: E402
from app.ocr.engine import OCREngine  # noqa: E402
from app.pdf.handler import PDFHandler  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
DEFAULT_CONFIG_PATH = Path(__file__).with_suffix(".yaml")


def resolve_path(path_value: str | None, base_dir: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_config() -> dict:
    config_path = Path(os.environ.get("EXTRACT_TEXT_CONFIG", DEFAULT_CONFIG_PATH)).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config["_config_path"] = str(config_path)
    config["_base_dir"] = str(config_path.parent)
    return config


def extract_from_pdf(input_path: Path, ocr_engine: OCREngine) -> tuple[list[PageResult], list[OCRResult]]:
    handler = PDFHandler(ocr_engine)
    return handler.process(str(input_path)), []


def extract_from_image(input_path: Path, ocr_engine: OCREngine) -> tuple[list[PageResult], list[OCRResult]]:
    raw_ocr, text = ocr_engine.extract(str(input_path))
    return [PageResult(page=0, text=text, method=ExtractionMethod.OCR)], raw_ocr


def format_text(input_path: Path, pages: list[PageResult]) -> str:
    sections: list[str] = [f"# Extracted text: {input_path.name}"]

    for page in pages:
        page_label = page.page + 1
        sections.append(f"\n--- Page {page_label} ({page.method.value}) ---\n")
        sections.append(page.text.strip() or "[no text extracted]")

    return "\n".join(sections).strip() + "\n"


def format_json(
    input_path: Path,
    pages: list[PageResult],
    raw_ocr: list[OCRResult],
    include_ocr_details: bool,
) -> str:
    payload = {
        "source": str(input_path),
        "pages": [page.model_dump(mode="json") for page in pages],
    }
    if include_ocr_details:
        payload["ocr_results"] = [result.model_dump(mode="json") for result in raw_ocr]
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def write_or_print(content: str, output_path: Path | None) -> None:
    if output_path is None:
        print(content, end="")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"Wrote extracted text to {output_path}")


def main() -> int:
    try:
        config = load_config()
    except Exception as e:
        print(f"Failed to load config: {e}", file=sys.stderr)
        return 1

    base_dir = Path(config["_base_dir"])
    input_path = resolve_path(config.get("input_path"), base_dir)
    output_path = resolve_path(config.get("output_path"), base_dir)
    output_format = str(config.get("output_format", "text")).lower()
    show_ocr_details = bool(config.get("show_ocr_details", False))

    if input_path is None:
        print("Config field 'input_path' is required.", file=sys.stderr)
        return 1

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    suffix = input_path.suffix.lower()
    if suffix != ".pdf" and suffix not in IMAGE_EXTENSIONS:
        print(f"Unsupported file type: {suffix}", file=sys.stderr)
        return 1

    if output_format not in {"text", "json"}:
        print("Config field 'output_format' must be 'text' or 'json'.", file=sys.stderr)
        return 1

    ocr_engine = OCREngine()
    try:
        if suffix == ".pdf":
            pages, raw_ocr = extract_from_pdf(input_path, ocr_engine)
        else:
            pages, raw_ocr = extract_from_image(input_path, ocr_engine)
    except Exception as e:
        print(f"Text extraction failed: {e}", file=sys.stderr)
        return 1

    if not pages:
        print(f"No pages were extracted from: {input_path}", file=sys.stderr)
        return 1

    if output_format == "json":
        content = format_json(
            input_path=input_path,
            pages=pages,
            raw_ocr=raw_ocr,
            include_ocr_details=show_ocr_details,
        )
    else:
        content = format_text(input_path, pages)

    write_or_print(content, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
