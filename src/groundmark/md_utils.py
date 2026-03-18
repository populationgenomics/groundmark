"""Markdown post-processing utilities."""

import collections
import re
from collections.abc import Sequence


def renumber_markers(markdown_chunks: Sequence[str]) -> list[str]:
    """Renumber ``<!--table-->`` and ``<!--figure-->`` markers across chunks.

    Transforms e.g. ``<!--table-->`` into ``<!--table: 1-->``, with counters
    running across all chunks so numbering is document-wide.
    """
    counters: collections.Counter[str] = collections.Counter()

    def _renumber(match: re.Match[str]) -> str:
        kind = match.group(1)
        counters[kind] += 1
        return f"<!--{kind}: {counters[kind]}-->"

    return [re.sub(r"<!--(table|figure)-->", _renumber, chunk) for chunk in markdown_chunks]
