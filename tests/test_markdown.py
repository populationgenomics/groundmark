"""Tests for PDF→Markdown generation (line-number stripping, NFKC normalization)."""

from unittest.mock import AsyncMock, patch

import pytest

from groundmark.convert import _agent, _generate_markdown


@patch.object(_agent, "run", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_calls_agent_with_chunk_data(mock_run: AsyncMock) -> None:
    mock_run.return_value.output = "1|# Hello\n2|\n3|Some text"

    result = await _generate_markdown(b"fake-pdf-bytes", "anthropic:claude-opus-4-6", "test prompt")

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    user_prompt = call_args[0][0]
    assert user_prompt[0].data == b"fake-pdf-bytes"
    assert user_prompt[0].media_type == "application/pdf"
    assert call_args[1]["model"] == "anthropic:claude-opus-4-6"

    # Line-number prefixes should be stripped.
    assert result == "# Hello\n\nSome text"


@patch.object(_agent, "run", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_strips_line_numbers(mock_run: AsyncMock) -> None:
    mock_run.return_value.output = "1|First line\n2|Second line\n3|\n4|Fourth"

    result = await _generate_markdown(b"fake-pdf-bytes", "anthropic:claude-opus-4-6", "test prompt")

    assert result == "First line\nSecond line\n\nFourth"


@patch.object(_agent, "run", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_nfkc_normalizes_output(mock_run: AsyncMock) -> None:
    # Superscript digits and ligatures should be decomposed to ASCII equivalents.
    mock_run.return_value.output = "1|overlap with NS\u00b9\u2070\u00b7\u00b9\u00b9 and \ufb01ndings"

    result = await _generate_markdown(b"fake-pdf-bytes", "anthropic:claude-opus-4-6", "test prompt")

    assert result == "overlap with NS10\u00b711 and findings"
