"""Tests for DocumentIndex quote resolution."""

from pathlib import Path

from groundmark import BBox, DocumentIndex

DATA_DIR = Path(__file__).parent / "data"
TWO_PAGES_PDF = (DATA_DIR / "two_pages.pdf").read_bytes()


class TestDocumentIndex:
    def test_extraction(self) -> None:
        doc = DocumentIndex(TWO_PAGES_PDF)
        assert len(doc._pages) == 2
        assert "first page" in doc._flat_str.lower()
        assert "second page" in doc._flat_str.lower()

    def test_resolve_exact_phrase(self) -> None:
        doc = DocumentIndex(TWO_PAGES_PDF)
        result = doc.resolve(["multiple sentences for alignment testing"])
        locs = result["multiple sentences for alignment testing"]
        assert len(locs) >= 1
        page, bbox = locs[0]
        assert page == 0
        assert isinstance(bbox, BBox)
        assert 0 <= bbox.top < bbox.bottom <= 1000
        assert 0 <= bbox.left < bbox.right <= 1000

    def test_resolve_on_second_page(self) -> None:
        doc = DocumentIndex(TWO_PAGES_PDF)
        result = doc.resolve(["screening using Covidence software"])
        locs = result["screening using Covidence software"]
        assert len(locs) >= 1
        assert locs[0][0] == 1  # page 1 (0-indexed)

    def test_resolve_batch(self) -> None:
        doc = DocumentIndex(TWO_PAGES_PDF)
        quotes = [
            "first page of the test document",
            "second page continues with different content",
        ]
        result = doc.resolve(quotes)
        assert len(result) == 2
        for q in quotes:
            assert len(result[q]) >= 1
        # First quote on page 0, second on page 1.
        assert result[quotes[0]][0][0] == 0
        assert result[quotes[1]][0][0] == 1

    def test_resolve_no_match(self) -> None:
        doc = DocumentIndex(TWO_PAGES_PDF)
        result = doc.resolve(["xyzzy plugh completely invented gibberish"])
        assert result["xyzzy plugh completely invented gibberish"] == []

    def test_resolve_empty_quotes(self) -> None:
        doc = DocumentIndex(TWO_PAGES_PDF)
        assert doc.resolve([]) == {}

    def test_resolve_reuse(self) -> None:
        """DocumentIndex can be reused for multiple resolve calls."""
        doc = DocumentIndex(TWO_PAGES_PDF)
        r1 = doc.resolve(["first page of the test document"])
        r2 = doc.resolve(["first page of the test document"])
        assert r1 == r2

    def test_resolve_custom_weights(self) -> None:
        doc = DocumentIndex(TWO_PAGES_PDF)
        result = doc.resolve(
            ["first page of the test document"],
            match=10,
            mismatch=-5,
            gap_open=-25,
            gap_extend=-1,
        )
        assert len(result["first page of the test document"]) >= 1
