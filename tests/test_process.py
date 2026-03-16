from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from anchorite import Anchor, BBox

from groundmark.process import Config, process

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def config() -> Config:
    return Config(model="anthropic:claude-opus-4-6")


@pytest.fixture
def simple_pdf_bytes() -> bytes:
    return (DATA_DIR / "simple_hello.pdf").read_bytes()


@patch("groundmark.parse.PdfplumberAnchorProvider.generate_anchors", new_callable=AsyncMock)
@patch("groundmark.markdown.PydanticAIMarkdownProvider.generate_markdown", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_process(
    mock_generate_md: AsyncMock,
    mock_generate_anchors: AsyncMock,
    config: Config,
    simple_pdf_bytes: bytes,
) -> None:
    mock_generate_md.return_value = "Hello\n\n"
    mock_generate_anchors.return_value = [
        Anchor(text="Hello", page=0, box=BBox(100, 100, 200, 200)),
    ]

    result = await process(simple_pdf_bytes, config)

    assert "Hello" in result.annotated_markdown
    assert "<span data-bbox=" in result.annotated_markdown
    assert result.coverage_percent > 0

    mock_generate_md.assert_called_once()
