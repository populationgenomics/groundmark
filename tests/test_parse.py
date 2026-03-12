from pathlib import Path

import pytest
from anchorite.document import chunks

from groundmark.parse import PdfplumberAnchorProvider

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def provider() -> PdfplumberAnchorProvider:
    return PdfplumberAnchorProvider()


@pytest.mark.asyncio
async def test_extracts_lines(provider: PdfplumberAnchorProvider) -> None:
    """Smoke test: basic line extraction works."""
    pdf_bytes = (DATA_DIR / "hello_world.pdf").read_bytes()

    chunk = next(chunks(pdf_bytes))
    anchors = await provider.generate_anchors(chunk)
    texts = {a.text for a in anchors}
    assert "Hello World" in texts
    assert "Second line" in texts


@pytest.mark.asyncio
async def test_table_rows_detected(provider: PdfplumberAnchorProvider) -> None:
    """Table cells on the same row should appear in a single anchor."""
    pdf_bytes = (DATA_DIR / "table_2x2.pdf").read_bytes()

    chunk = next(chunks(pdf_bytes))
    anchors = await provider.generate_anchors(chunk)
    texts = {a.text for a in anchors}

    assert any("Cell A" in t and "Cell B" in t for t in texts), f"Row 1 not merged: {texts}"
    assert any("Cell C" in t and "Cell D" in t for t in texts), f"Row 2 not merged: {texts}"
