"""Quote-only alignment: resolve verbatim quotes to PDF bounding boxes.

Given raw PDF bytes and a list of quote strings, extracts per-character
bounding boxes via pypdfium2, then uses Smith-Waterman local-global alignment
(via seq_smith) to locate each quote in the document text and return the
corresponding bounding boxes.
"""

import logging
import string
from collections.abc import Sequence

import numpy as np
import pypdfium2 as pdfium
import seq_smith

from .pdf_chars import Char, build_char_index, extract_page_chars, line_bboxes
from .types import BBox

logger = logging.getLogger(__name__)

# Default alignment weights (cur-ai-ss-style, scaled x20 for seq_smith integer compatibility).
# match=+1, mismatch=-0.5, gap_open=-2.5, gap_extend=-0.05  →  x20  →  20, -10, -50, -1
DEFAULT_MATCH = 20
DEFAULT_MISMATCH = -10
DEFAULT_GAP_OPEN = -50
DEFAULT_GAP_EXTEND = -1

# Minimum alignment score to accept a match (≈15 matched chars x 20).
_MIN_ALIGNMENT_SCORE = 300
# Coverage thresholds (fraction of normalized quote that must be matched).
_WARN_COVERAGE = 0.5
_FAIL_COVERAGE = 0.3
# Maximum iterations for mask-and-realign (handles split/multi-region quotes).
_MAX_ITERATIONS = 10

# Normalization alphabet for seq_smith encoding.
_MASK_CHAR = "#"
_ALIGN_ALPHABET = string.ascii_lowercase + string.digits + " " + _MASK_CHAR


def _build_score_matrix(match: int, mismatch: int) -> np.ndarray:
    """Build a score matrix with mask-char support."""
    sm = seq_smith.make_score_matrix(_ALIGN_ALPHABET, match, mismatch)
    mask_byte = seq_smith.encode(_MASK_CHAR, _ALIGN_ALPHABET)[0]
    sm[mask_byte, :] = -100
    sm[:, mask_byte] = -100
    return sm


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

    __slots__ = ("chars", "height", "width")

    def __init__(self, chars: list[Char], width: float, height: float) -> None:
        self.chars = chars
        self.width = width
        self.height = height


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
            self._pages.append(_PageData(chars, page.get_width(), page.get_height()))

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

    def _resolve_single(
        self,
        score_matrix: np.ndarray,
        gap_open: int,
        gap_extend: int,
        quote: str,
    ) -> list[tuple[int, BBox]]:
        """Resolve a single quote using iterative mask-and-realign."""
        clean_quote = quote.strip()
        if not clean_quote:
            return []

        norm_quote = _normalize(clean_quote)
        if not norm_quote:
            return []

        mask_byte = seq_smith.encode(_MASK_CHAR, _ALIGN_ALPHABET)[0]
        current_norm = bytearray(self._flat_norm)
        total_len = len(norm_quote)
        matched_len = 0
        all_page_chars: dict[int, list[Char]] = {}

        for _ in range(_MAX_ITERATIONS):
            aln = seq_smith.local_global_align(
                bytes(current_norm),
                norm_quote,
                score_matrix,
                gap_open,
                gap_extend,
            )
            if aln.score < _MIN_ALIGNMENT_SCORE:
                break

            iteration_matched = 0
            for frag in aln.fragments:
                if frag.fragment_type == seq_smith.FragmentType.Match:
                    # Mask consumed portion of document to prevent re-matching.
                    for j in range(frag.sa_start, frag.sa_start + frag.len):
                        current_norm[j] = mask_byte

                    # Map normalized positions back to flat-string positions.
                    flat_start = self._norm_to_flat[frag.sa_start]
                    flat_end = self._norm_to_flat[frag.sa_start + frag.len]

                    # Collect matched characters grouped by page.
                    for page_idx, chars in self._chars_for_flat_range(flat_start, flat_end).items():
                        all_page_chars.setdefault(page_idx, []).extend(chars)

                    iteration_matched += frag.len
                elif frag.fragment_type == seq_smith.FragmentType.AGap:
                    # Document gap — mask to avoid re-matching but don't collect chars.
                    for j in range(frag.sa_start, frag.sa_start + frag.len):
                        current_norm[j] = mask_byte
                    iteration_matched += frag.len

            if iteration_matched == 0:
                break
            matched_len += iteration_matched

        # Coverage check.
        if total_len > 0 and matched_len < total_len * _WARN_COVERAGE:
            logger.warning(
                "Low coverage for quote alignment: %d/%d for quote '%s'",
                matched_len,
                total_len,
                quote,
            )
            if matched_len < total_len * _FAIL_COVERAGE:
                return []

        # Convert collected characters to line-level bounding boxes.
        # page_idx is 0-based (array index), but the public API returns
        # 1-based page numbers to match PDF conventions (PDF spec, pdf.js,
        # react-pdf all use 1-based pages).
        results: list[tuple[int, BBox]] = []
        for page_idx in sorted(all_page_chars):
            pd = self._pages[page_idx]
            boxes = line_bboxes(all_page_chars[page_idx], pd.width, pd.height)
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
    ) -> dict[str, list[tuple[int, BBox]]]:
        """Resolve verbatim quotes to bounding boxes.

        Aligns each quote against the full document text using Smith-Waterman
        local-global alignment. Matched characters are clustered into
        line-level bounding boxes.

        Args:
            quotes: Verbatim strings to locate in the PDF.
            match: Score for matching characters (default 20, i.e. +1.0 scaled x20).
            mismatch: Score for mismatching characters (default -10, i.e. -0.5 scaled x20).
            gap_open: Penalty for opening a gap (default -50, i.e. -2.5 scaled x20).
            gap_extend: Penalty for extending a gap (default -1, i.e. -0.05 scaled x20).

        Returns:
            Mapping of quote string → list of ``(page, BBox)`` tuples. Pages are
            1-indexed (first page is 1). Quotes that cannot be matched with
            sufficient confidence return an empty list.
        """
        if not quotes or not self._flat_str:
            return {q: [] for q in quotes}

        score_matrix = _build_score_matrix(match, mismatch)

        return {quote: self._resolve_single(score_matrix, gap_open, gap_extend, quote) for quote in quotes}
