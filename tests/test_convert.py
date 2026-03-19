"""Tests for the convert module."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from groundmark.convert import Config, ConvertResult, convert

DATA_DIR = Path(__file__).parent / "data"
TWO_PAGES_PDF = (DATA_DIR / "two_pages.pdf").read_bytes()


@patch("groundmark.convert._generate_markdown", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_convert_single_chunk(mock_generate_md: AsyncMock) -> None:
    mock_generate_md.return_value = "# Hello\n\nSome text"

    result = await convert(TWO_PAGES_PDF, Config(model="anthropic:claude-opus-4-6"))

    assert isinstance(result, ConvertResult)
    assert "Hello" in result.markdown
    assert "<span" not in result.markdown
    mock_generate_md.assert_called_once()


@patch("groundmark.convert._generate_markdown", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_convert_multi_chunk(mock_generate_md: AsyncMock) -> None:
    mock_generate_md.side_effect = ["# Page 1", "# Page 2"]

    result = await convert(TWO_PAGES_PDF, Config(model="anthropic:claude-opus-4-6", page_count=1))

    assert "<!--page-->" in result.markdown
    assert "Page 1" in result.markdown
    assert "Page 2" in result.markdown


@pytest.mark.asyncio
async def test_convert_pre_generated_markdown() -> None:
    result = await convert(TWO_PAGES_PDF, Config(model="unused"), markdown="pre-generated content")
    assert result.markdown == "pre-generated content"
