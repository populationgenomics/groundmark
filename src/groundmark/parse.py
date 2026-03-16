"""Bounding box extraction from PDFs using pdfplumber."""

import io
import logging
import unicodedata

import pdfplumber
from anchorite import Anchor, BBox
from anchorite.document import DocumentChunk


class PdfplumberAnchorProvider:
    """AnchorProvider that extracts line-level bounding boxes via pdfplumber.

    Body text lines are emitted individually; table rows are merged into
    single anchors so that repeated short cell values (e.g. "Pathogenic")
    become unique when combined with the full row.

    Coordinates are normalized to a 0-1000 scale with top-left origin.
    """

    async def generate_anchors(self, chunk: DocumentChunk) -> list[Anchor]:
        pdf = pdfplumber.open(io.BytesIO(chunk.data))
        anchors: list[Anchor] = []

        for page_offset, page in enumerate(pdf.pages):
            pw, ph = page.width, page.height
            if pw <= 0 or ph <= 0:
                logging.warning(
                    "Page %d has invalid dimensions (%s x %s), skipping",
                    chunk.start_page + page_offset,
                    pw,
                    ph,
                )
                continue

            page_num = chunk.start_page + page_offset

            # Detect table regions on this page.
            table_bboxes = [table.bbox for table in page.find_tables()]

            lines = page.extract_text_lines(return_chars=False, use_text_flow=True)

            for line in lines:
                x0, x1 = line["x0"], line["x1"]
                # pdfplumber's top starts at baseline minus font size,
                # missing the ascender portion above the glyph. Pad upward
                # by ~30% of line height to approximate full glyph bounds.
                line_h = line["bottom"] - line["top"]
                y0 = line["top"] - 0.3 * line_h
                y1 = line["bottom"]
                text = unicodedata.normalize("NFKC", line["text"].strip())
                if not text:
                    continue

                in_table = any(_rects_intersect((x0, y0, x1, y1), tb) for tb in table_bboxes)

                if in_table:
                    # Table lines are already row-level from extract_text_lines.
                    anchors.append(
                        Anchor(
                            page=page_num,
                            box=_normalize(x0, y0, x1, y1, pw, ph),
                            text=text,
                        )
                    )
                else:
                    anchors.append(
                        Anchor(
                            page=page_num,
                            box=_normalize(x0, y0, x1, y1, pw, ph),
                            text=text,
                        )
                    )

        pdf.close()
        logging.debug("Generated %d bounding boxes", len(anchors))
        return anchors


def _rects_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """Check if two (x0, y0, x1, y1) rectangles overlap."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


def _normalize(x0: float, y0: float, x1: float, y1: float, pw: float, ph: float) -> BBox:
    """Convert PDF coordinates to 0-1000 scale with top-left origin."""
    return BBox(
        top=int(y0 / ph * 1000),
        left=int(x0 / pw * 1000),
        bottom=int(y1 / ph * 1000),
        right=int(x1 / pw * 1000),
    )
