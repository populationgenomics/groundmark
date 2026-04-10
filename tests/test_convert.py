"""Tests for the convert module."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from groundmark.convert import Config, ConvertResult, _ChunkResult, convert

DATA_DIR = Path(__file__).parent / "data"
TWO_PAGES_PDF = (DATA_DIR / "two_pages.pdf").read_bytes()


@patch("groundmark.convert._generate_markdown", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_convert_single_chunk(mock_generate_md: AsyncMock) -> None:
    mock_generate_md.return_value = _ChunkResult(markdown="# Hello\n\nSome text", all_messages=[])

    result = await convert(TWO_PAGES_PDF, Config(model="anthropic:claude-opus-4-6"))

    assert isinstance(result, ConvertResult)
    assert "Hello" in result.markdown
    assert "<span" not in result.markdown
    assert result.all_messages == [[]]
    mock_generate_md.assert_called_once()


@patch("groundmark.convert._generate_markdown", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_convert_multi_chunk(mock_generate_md: AsyncMock) -> None:
    mock_generate_md.side_effect = [
        _ChunkResult(markdown="# Page 1", all_messages=[{"kind": "request"}]),
        _ChunkResult(markdown="# Page 2", all_messages=[{"kind": "request"}]),
    ]

    result = await convert(TWO_PAGES_PDF, Config(model="anthropic:claude-opus-4-6", page_count=1))

    assert "<!--page-->" in result.markdown
    assert "Page 1" in result.markdown
    assert "Page 2" in result.markdown
    assert len(result.all_messages) == 2


@pytest.mark.asyncio
async def test_convert_pre_generated_markdown() -> None:
    result = await convert(TWO_PAGES_PDF, Config(model="unused"), markdown="pre-generated content")
    assert result.markdown == "pre-generated content"
    assert result.all_messages == []
