"""PDF chunking utilities."""

import io
from collections.abc import Iterator

import pypdfium2 as pdfium


def chunks(
    pdf_bytes: bytes,
    *,
    page_count: int | None = None,
) -> Iterator[tuple[bytes, int, int]]:
    """Split a PDF into byte chunks by page count.

    Args:
        pdf_bytes: Raw PDF file bytes.
        page_count: Number of pages per chunk. If ``None``, the entire PDF is
            yielded as a single chunk.

    Yields:
        ``(chunk_bytes, start_page, end_page)`` tuples. ``end_page`` is exclusive.
    """
    doc = pdfium.PdfDocument(pdf_bytes)
    doc_page_count = len(doc)
    if page_count is None:
        yield pdf_bytes, 0, doc_page_count
        return

    for start_page in range(0, doc_page_count, page_count):
        end_page = min(start_page + page_count, doc_page_count)
        new_doc = pdfium.PdfDocument.new()
        new_doc.import_pages(doc, list(range(start_page, end_page)))
        buf = io.BytesIO()
        new_doc.save(buf)
        yield buf.getvalue(), start_page, end_page
