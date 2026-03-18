"""Per-character bounding box extraction and clustering from PDF pages.

Uses pypdfium2 to extract individual characters with their positions, then
clusters them into line-level bounding boxes using y-overlap grouping.
"""

import dataclasses
import math
import unicodedata
from typing import NamedTuple

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from .types import BBox

# Character normalization map: ligatures, smart quotes, dashes, etc.
_CHAR_NORM: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201a": ",",
    "\u2013": "-",
    "\u2014": "--",
    "\u2212": "-",
    "\u2010": "-",
    "\u2011": "-",
    "\u00ad": "",
    "\u00a0": " ",
    "\ufffe": "",
}


@dataclasses.dataclass(frozen=True)
class Char:
    """A single extracted character with its bounding box and font size."""

    text: str
    """Normalized character(s)."""
    x0: float
    """Left edge in PDF coordinates (pts)."""
    y0: float
    """Bottom edge in PDF coordinates (pts, origin bottom-left)."""
    x1: float
    """Right edge in PDF coordinates (pts)."""
    y1: float
    """Top edge in PDF coordinates (pts)."""
    font_size: float
    """Scaled font size."""


def extract_page_chars(page: pdfium.PdfPage) -> list[Char]:
    """Extract non-whitespace characters with bounding boxes from a PDF page.

    Handles surrogate pairs for non-BMP characters (e.g. Mathematical Italic
    symbols U+1D400-U+1D7FF) and normalizes ligatures, smart quotes, and
    dashes via NFKC normalization.
    """
    textpage = page.get_textpage()
    total_chars = textpage.count_chars()
    chars: list[Char] = []
    char_index = 0

    for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_TEXT]):
        buf_size = pdfium_c.FPDFTextObj_GetText(obj, textpage, None, 0)
        buf = (pdfium_c.FPDF_WCHAR * buf_size)()
        pdfium_c.FPDFTextObj_GetText(obj, textpage, buf, buf_size)
        obj_text = bytes(buf).decode("utf-16-le").rstrip("\x00")

        m = obj.get_matrix()
        font_size = obj.get_font_size() * math.sqrt(m.a**2 + m.b**2)

        obj_pos = 0
        while obj_pos < len(obj_text) and char_index < total_chars:
            cp = pdfium_c.FPDFText_GetUnicode(textpage, char_index)
            # PDFium counts non-BMP characters as two UTF-16 surrogate-pair
            # indices. Detect a high surrogate and reassemble the full code point.
            if 0xD800 <= cp <= 0xDBFF:
                if char_index + 1 < total_chars:
                    cp_low = pdfium_c.FPDFText_GetUnicode(textpage, char_index + 1)
                    if 0xDC00 <= cp_low <= 0xDFFF:
                        cp = 0x10000 + (cp - 0xD800) * 0x400 + (cp_low - 0xDC00)
                        ci_for_box = char_index
                        char_index += 2
                        obj_pos += 1
                    else:
                        char_index += 1
                        obj_pos += 1
                        continue
                else:
                    char_index += 1
                    obj_pos += 1
                    continue
            elif 0xDC00 <= cp <= 0xDFFF:
                # Orphaned low surrogate — skip.
                char_index += 1
                obj_pos += 1
                continue
            else:
                ci_for_box = char_index
                char_index += 1

            text = chr(cp)
            if text in ("\r", "\n"):
                # Line-break markers inserted by PDFium are absent from obj_text.
                continue  # char_index already advanced; do NOT advance obj_pos
            obj_pos += 1

            if not text.isspace():
                normalized = _CHAR_NORM.get(text, text)
                # Map Mathematical Alphanumeric Symbols and other compatibility
                # characters to ASCII equivalents via NFKC normalization.
                normalized = unicodedata.normalize("NFKC", normalized)
                if normalized:
                    left, bottom, right, top = textpage.get_charbox(ci_for_box, loose=False)
                    if right > left and top > bottom:
                        chars.append(Char(normalized, left, bottom, right, top, font_size))

    return chars


class CharIndex(NamedTuple):
    """A flat text string built from page characters, with position mapping."""

    flat_str: str
    """Concatenated character text with inter-word spaces."""
    flat_to_char: list[int]
    """flat_to_char[i] = index into the source char list for flat_str[i]."""


def build_char_index(chars: list[Char]) -> CharIndex:
    """Build a flat string and per-position index mapping back to characters.

    Inserts spaces between characters when the horizontal gap exceeds 20% of
    the font size, approximating word boundaries.
    """
    parts: list[str] = []
    flat_to_char: list[int] = []

    for i, ch in enumerate(chars):
        for c in ch.text:
            parts.append(c)
            flat_to_char.append(i)
        if i + 1 < len(chars):
            gap = chars[i + 1].x0 - ch.x1
            if gap > ch.font_size * 0.2:
                parts.append(" ")
                flat_to_char.append(i)

    return CharIndex("".join(parts), flat_to_char)


def bbox_from_chars(chars: list[Char], page_width: float, page_height: float) -> BBox:
    """Convert a list of characters to a single BBox in 0-1000 normalized coordinates."""
    x0 = min(c.x0 for c in chars)
    y0 = min(c.y0 for c in chars)
    x1 = max(c.x1 for c in chars)
    y1 = max(c.y1 for c in chars)
    top = round((1.0 - y1 / page_height) * 1000)
    left = round(x0 / page_width * 1000)
    bottom = round((1.0 - y0 / page_height) * 1000)
    right = round(x1 / page_width * 1000)
    return BBox(top=top, left=left, bottom=bottom, right=right)


def line_bboxes(chars: list[Char], page_width: float, page_height: float) -> list[BBox]:
    """Return one BBox per visual line of characters.

    Characters are sorted top-to-bottom by y-midpoint and grouped into lines
    by y-overlap: a character joins the current line when its y-range overlaps
    the accumulated line band; otherwise it starts a new line. This produces
    one tight box per text line regardless of font size or column layout.
    """
    if not chars:
        return []

    # Sort top-to-bottom (descending y in PDF coords where y increases upward).
    by_y = sorted(chars, key=lambda c: -(c.y0 + c.y1) / 2)

    clusters: list[list[Char]] = [[by_y[0]]]
    band_y0 = by_y[0].y0
    band_y1 = by_y[0].y1

    for ch in by_y[1:]:
        overlap = min(ch.y1, band_y1) - max(ch.y0, band_y0)
        if overlap > 0:
            clusters[-1].append(ch)
            band_y0 = min(band_y0, ch.y0)
            band_y1 = max(band_y1, ch.y1)
        else:
            clusters.append([ch])
            band_y0 = ch.y0
            band_y1 = ch.y1

    return [bbox_from_chars(cluster, page_width, page_height) for cluster in clusters]
