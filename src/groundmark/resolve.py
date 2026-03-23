"""Quote-only alignment: resolve verbatim quotes to PDF bounding boxes.

Given raw PDF bytes and a list of quote strings, extracts per-character
bounding boxes via pypdfium2, then uses Smith-Waterman local-global alignment
(via seq_smith) to locate each quote in the document text and return the
corresponding bounding boxes.
"""

import string
from collections.abc import Sequence

import numpy as np
import pypdfium2 as pdfium
import seq_smith

from .pdf_chars import Char, build_char_index, extract_page_chars, line_bboxes
from .types import BBox

# Default alignment weights (anchorite-style integers for seq_smith).
DEFAULT_MATCH = 1
DEFAULT_MISMATCH = -1
DEFAULT_GAP_OPEN = -2
DEFAULT_GAP_EXTEND = -2

# Minimum alignment score to accept a match (≈15 matched chars).
_MIN_ALIGNMENT_SCORE = 15

# Normalization alphabet for seq_smith encoding.
_ALIGN_ALPHABET = string.ascii_lowercase + string.digits + " "


def _build_score_matrix(match: int, mismatch: int) -> np.ndarray:
    """Build a score matrix for alignment."""
    return seq_smith.make_score_matrix(_ALIGN_ALPHABET, match, mismatch)


def _normalize(text: str) -> bytes:
    """Lowercase + collapse non-alphanumeric runs to a single space, then encode."""
    result: list[str] = []
    for c in text:
        lc = c.lower()
        if lc in string.ascii_letters + string.digits:
            result.append(lc)
        elif result and result[-1] != " ":
            result.append(" ")
    return seq_smith.encode("".join(result), _ALIGN_ALPHABET)


def _build_norm_to_flat(flat_str: str) -> list[int]:
    """Build a mapping from normalized-string positions to flat-string positions.

    Must mirror ``_normalize`` logic exactly so positions correspond.
    """
    mapping: list[int] = []
    last_was_space = False
    for i, c in enumerate(flat_str):
        lc = c.lower()
        if lc in string.ascii_letters + string.digits:
            mapping.append(i)
            last_was_space = False
        elif not last_was_space:
            mapping.append(i)
            last_was_space = True
    # Sentinel for exclusive end lookups.
    mapping.append(len(flat_str))
    return mapping


class _PageData:
    """Per-page extracted character data and dimensions."""

    __slots__ = ("chars", "height", "origin_x", "origin_y", "width")

    def __init__(self, chars: list[Char], width: float, height: float, origin_x: float, origin_y: float) -> None:
        self.chars = chars
        self.width = width
        self.height = height
        self.origin_x = origin_x
        self.origin_y = origin_y


class DocumentIndex:
    """Pre-extracted character data from a PDF, ready for quote resolution.

    Construction extracts per-character bounding boxes from every page using
    pypdfium2 and builds a document-level flat string. This is the expensive
    step. Once built, call :meth:`resolve` cheaply as many times as needed.

    Example::

        doc = DocumentIndex(pdf_bytes)
        result1 = doc.resolve(["first quote", "second quote"])
        # later, with new quotes against the same PDF:
        result2 = doc.resolve(["third quote"])
    """

    def __init__(self, pdf_bytes: bytes) -> None:
        doc = pdfium.PdfDocument(pdf_bytes)
        self._pages: list[_PageData] = []

        flat_parts: list[str] = []
        self._flat_to_page: list[int] = []
        self._flat_to_page_char: list[int] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            chars = extract_page_chars(page)
            ci = build_char_index(chars)
            mb = page.get_mediabox()
            self._pages.append(_PageData(chars, page.get_width(), page.get_height(), mb[0], mb[1]))

            if flat_parts:
                # Space separator between pages (prevents cross-page token merging).
                flat_parts.append(" ")
                self._flat_to_page.append(page_idx)
                self._flat_to_page_char.append(-1)

            flat_parts.append(ci.flat_str)
            self._flat_to_page.extend([page_idx] * len(ci.flat_str))
            self._flat_to_page_char.extend(ci.flat_to_char)

        self._flat_str = "".join(flat_parts)
        self._flat_norm = _normalize(self._flat_str)
        self._norm_to_flat = _build_norm_to_flat(self._flat_str)

    def _chars_for_flat_range(self, flat_start: int, flat_end: int) -> dict[int, list[Char]]:
        """Map a flat-string range to page-grouped character lists."""
        page_chars: dict[int, list[Char]] = {}
        seen: set[tuple[int, int]] = set()
        for i in range(flat_start, min(flat_end, len(self._flat_to_page))):
            page_idx = self._flat_to_page[i]
            char_idx = self._flat_to_page_char[i]
            if char_idx < 0:
                continue  # page separator
            key = (page_idx, char_idx)
            if key in seen:
                continue
            seen.add(key)
            pd = self._pages[page_idx]
            if char_idx < len(pd.chars):
                page_chars.setdefault(page_idx, []).append(pd.chars[char_idx])
        return page_chars

    def _bboxes_from_alignment(self, aln: seq_smith.Alignment) -> list[tuple[int, BBox]]:
        """Convert an alignment result to page-grouped bounding boxes."""
        if aln.score < _MIN_ALIGNMENT_SCORE:
            return []

        # Collect matched characters grouped by page.
        all_page_chars: dict[int, list[Char]] = {}
        for frag in aln.fragments:
            if frag.fragment_type == seq_smith.FragmentType.Match:
                flat_start = self._norm_to_flat[frag.sa_start]
                flat_end = self._norm_to_flat[frag.sa_start + frag.len]
                for page_idx, chars in self._chars_for_flat_range(flat_start, flat_end).items():
                    all_page_chars.setdefault(page_idx, []).extend(chars)

        # Convert collected characters to line-level bounding boxes.
        # page_idx is 0-based (array index), but the public API returns
        # 1-based page numbers to match PDF conventions (PDF spec, pdf.js,
        # react-pdf all use 1-based pages).
        results: list[tuple[int, BBox]] = []
        for page_idx in sorted(all_page_chars):
            pd = self._pages[page_idx]
            boxes = line_bboxes(all_page_chars[page_idx], pd.width, pd.height, pd.origin_x, pd.origin_y)
            results.extend((page_idx + 1, box) for box in boxes)

        return results

    def resolve(
        self,
        quotes: Sequence[str],
        *,
        match: int = DEFAULT_MATCH,
        mismatch: int = DEFAULT_MISMATCH,
        gap_open: int = DEFAULT_GAP_OPEN,
        gap_extend: int = DEFAULT_GAP_EXTEND,
        num_threads: int | None = None,
    ) -> dict[str, list[tuple[int, BBox]]]:
        """Resolve verbatim quotes to bounding boxes.

        Aligns each quote against the full document text using Smith-Waterman
        local-global alignment. Matched characters are clustered into
        line-level bounding boxes.

        Args:
            quotes: Verbatim strings to locate in the PDF.
            match: Score for matching characters.
            mismatch: Score for mismatching characters.
            gap_open: Penalty for opening a gap.
            gap_extend: Penalty for extending a gap.
            num_threads: Thread count for batch alignment. ``None`` lets
                seq_smith choose a default.

        Returns:
            Mapping of quote string → list of ``(page, BBox)`` tuples. Pages are
            1-indexed (first page is 1). Quotes that cannot be matched with
            sufficient confidence return an empty list.
        """
        if not quotes or not self._flat_str:
            return {q: [] for q in quotes}

        score_matrix = _build_score_matrix(match, mismatch)

        # Normalize all quotes upfront, tracking which are non-empty.
        norm_quotes: list[bytes] = []
        quote_indices: list[int] = []  # index into *quotes* for each norm entry
        for i, quote in enumerate(quotes):
            clean = quote.strip()
            if not clean:
                continue
            nq = _normalize(clean)
            if nq:
                norm_quotes.append(nq)
                quote_indices.append(i)

        results: dict[str, list[tuple[int, BBox]]] = {q: [] for q in quotes}

        if not norm_quotes:
            return results

        alignments = seq_smith.local_global_align_many(
            self._flat_norm,
            norm_quotes,
            score_matrix,
            gap_open,
            gap_extend,
            num_threads=num_threads,
        )

        for idx, aln in zip(quote_indices, alignments, strict=True):
            results[quotes[idx]] = self._bboxes_from_alignment(aln)

        return results
