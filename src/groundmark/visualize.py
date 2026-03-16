"""Debug visualizer: overlay bounding boxes from annotated Markdown onto the source PDF."""

import asyncio
import io
import logging
import re
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from anchorite import Anchor, align, annotate
from anchorite.document import chunks
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Highlight
from pypdf.generic import ArrayObject, FloatObject

from groundmark.markdown import PydanticAIMarkdownProvider
from groundmark.parse import PdfplumberAnchorProvider

_SPAN_RE = re.compile(
    r'<span\s+data-bbox="(\d+,\d+,\d+,\d+(?:;\d+,\d+,\d+,\d+)*)"\s+data-page="(\d+)">(.*?)</span>',
    re.DOTALL,
)

app = typer.Typer()


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _add_highlight(
    writer: PdfWriter,
    page_num: int,
    bbox: tuple[int, int, int, int],
    page_width: float,
    page_height: float,
    *,
    color: tuple[float, float, float],
) -> None:
    top, left, bottom, right = bbox
    # Convert from 0-1000 scale to PDF coordinates (bottom-left origin).
    x0 = left / 1000 * page_width
    x1 = right / 1000 * page_width
    y0 = page_height - bottom / 1000 * page_height
    y1 = page_height - top / 1000 * page_height

    hex_color = "".join(f"{int(c * 255):02x}" for c in color)
    quad_points = ArrayObject(
        [
            FloatObject(x0),
            FloatObject(y1),  # top-left
            FloatObject(x1),
            FloatObject(y1),  # top-right
            FloatObject(x0),
            FloatObject(y0),  # bottom-left
            FloatObject(x1),
            FloatObject(y0),  # bottom-right
        ]
    )
    annotation = Highlight(
        rect=(x0, y0, x1, y1),
        quad_points=quad_points,
        highlight_color=hex_color,
    )
    writer.add_annotation(page_number=page_num, annotation=annotation)


def _overlay_bboxes(
    pdf_bytes: bytes,
    annotated_markdown: str,
    raw_anchors: list[Anchor],
) -> bytes:
    """Draw bounding box highlights onto the PDF.

    Blue = raw extracted boxes, Red = aligned (from annotated Markdown).
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)

    # Blue: raw extracted boxes.
    for anchor in raw_anchors:
        if anchor.page >= len(reader.pages):
            continue
        page = reader.pages[anchor.page]
        pw = float(page.mediabox.width)
        ph = float(page.mediabox.height)
        for box in anchor.boxes:
            _add_highlight(writer, anchor.page, (box.top, box.left, box.bottom, box.right), pw, ph, color=(0.8, 0.9, 1))

    # Red: aligned boxes from annotated Markdown.
    for match in _SPAN_RE.finditer(annotated_markdown):
        bbox_str = match[1]
        page_num = int(match[2])
        if page_num >= len(reader.pages):
            continue
        page = reader.pages[page_num]
        pw = float(page.mediabox.width)
        ph = float(page.mediabox.height)
        for group in bbox_str.split(";"):
            top, left, bottom, right = (int(v) for v in group.split(","))
            _add_highlight(writer, page_num, (top, left, bottom, right), pw, ph, color=(1, 0.85, 0.85))

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@app.command()
def visualize(
    input_pdf: Annotated[Path, typer.Argument(help="Path to the source PDF.")],
    output_pdf: Annotated[Path, typer.Argument(help="Path for the output PDF with bbox overlays.")],
    model: Annotated[str, typer.Option(help="Pydantic AI model string.")] = "",
    markdown_file: Annotated[Path | None, typer.Option("--markdown", "-m", help="Cached Markdown file.")] = None,
    uniqueness_threshold: Annotated[float, typer.Option("--threshold", "-t")] = 0.5,
    min_overlap: Annotated[float, typer.Option("--overlap", "-o")] = 0.9,
    page_count: Annotated[int | None, typer.Option("--page-count", "-p", help="Pages per chunk.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
) -> None:
    """Run the groundmark pipeline on a PDF and overlay bounding boxes."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)

    if not markdown_file and not model:
        _log("Error: provide --model or --markdown")
        raise SystemExit(1)

    asyncio.run(
        _run(
            input_pdf,
            output_pdf,
            model,
            markdown_file,
            uniqueness_threshold,
            min_overlap,
            page_count,
        )
    )


async def _run(
    input_pdf: Path,
    output_pdf: Path,
    model: str,
    markdown_file: Path | None,
    uniqueness_threshold: float,
    min_overlap: float,
    page_count: int | None,
) -> None:
    pdf_bytes = input_pdf.read_bytes()
    doc_chunks = list(chunks(pdf_bytes, page_count=page_count))
    t0 = time.perf_counter()

    # Extract anchors (bounding boxes) from the PDF.
    _log("Extracting bounding boxes...")
    t = time.perf_counter()
    provider = PdfplumberAnchorProvider()
    all_anchors = await asyncio.gather(*(provider.generate_anchors(c) for c in doc_chunks))
    flat_anchors = [a for chunk_anchors in all_anchors for a in chunk_anchors]
    _log(f"  {len(flat_anchors)} bounding boxes ({time.perf_counter() - t:.1f}s)")

    # Generate or load Markdown.
    if markdown_file:
        markdown = markdown_file.read_text()
        _log(f"  Loaded cached Markdown from {markdown_file}")
    else:
        _log("Generating Markdown...")
        t = time.perf_counter()
        md_provider = PydanticAIMarkdownProvider(model)
        md_chunks = await asyncio.gather(*(md_provider.generate_markdown(c) for c in doc_chunks))
        markdown = "\n\n<!--page-->\n\n".join(md_chunks)
        _log(f"  {len(markdown)} chars ({time.perf_counter() - t:.1f}s)")

    # Write plain markdown before alignment so it's available even if alignment hangs.
    if not markdown_file:
        md_path = output_pdf.with_suffix(".md")
        md_path.write_text(markdown)
        _log(f"  Markdown written to {md_path}")

    # Align anchors to Markdown.
    _log("Aligning...")
    t = time.perf_counter()
    alignment = align(
        flat_anchors,
        markdown,
        uniqueness_threshold=uniqueness_threshold,
        min_overlap=min_overlap,
    )
    annotated_markdown = annotate(markdown, alignment)
    coverage = sum(e - s for s, e in alignment.values()) / len(markdown) if markdown else 0.0
    _log(f"  {coverage:.1%} coverage ({time.perf_counter() - t:.1f}s)")

    output_bytes = _overlay_bboxes(pdf_bytes, annotated_markdown, flat_anchors)
    output_pdf.write_bytes(output_bytes)
    _log(f"  Output written to {output_pdf}")
    _log(f"Total: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    app()
