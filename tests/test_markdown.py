"""Tests for the PydanticAIMarkdownProvider."""

from unittest.mock import AsyncMock, patch

import pytest

from groundmark.document import DocumentChunk
from groundmark.markdown import PydanticAIMarkdownProvider, _agent


@pytest.fixture
def chunk() -> DocumentChunk:
    return DocumentChunk(
        document_sha256="abc",
        start_page=0,
        end_page=1,
        data=b"fake-pdf-bytes",
        mime_type="application/pdf",
    )


@patch.object(_agent, "run", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_calls_agent_with_chunk_data(mock_run: AsyncMock, chunk: DocumentChunk) -> None:
    mock_run.return_value.output = "1|# Hello\n2|\n3|Some text"

    provider = PydanticAIMarkdownProvider(model="anthropic:claude-opus-4-6")
    result = await provider.generate_markdown(chunk)

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
async def test_strips_line_numbers(mock_run: AsyncMock, chunk: DocumentChunk) -> None:
    mock_run.return_value.output = "1|First line\n2|Second line\n3|\n4|Fourth"

    provider = PydanticAIMarkdownProvider(model="anthropic:claude-opus-4-6")
    result = await provider.generate_markdown(chunk)

    assert result == "First line\nSecond line\n\nFourth"


@patch.object(_agent, "run", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_nfkc_normalizes_output(mock_run: AsyncMock, chunk: DocumentChunk) -> None:
    # Superscript digits and ligatures should be decomposed to ASCII equivalents.
    mock_run.return_value.output = "1|overlap with NS\u00b9\u2070\u00b7\u00b9\u00b9 and \ufb01ndings"

    provider = PydanticAIMarkdownProvider(model="anthropic:claude-opus-4-6")
    result = await provider.generate_markdown(chunk)

    assert result == "overlap with NS10\u00b711 and findings"
