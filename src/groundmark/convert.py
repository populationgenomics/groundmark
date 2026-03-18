"""PDF to Markdown conversion via LLM."""

import asyncio
import dataclasses

from .document import chunks
from .markdown import PydanticAIMarkdownProvider
from .md_utils import renumber_markers


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for PDF→Markdown conversion."""

    model: str
    """Pydantic AI model string (e.g. "bedrock:au.anthropic.claude-sonnet-4-6")."""
    page_count: int | None = None
    """Pages per chunk (None = whole PDF in one chunk)."""
    prompt: str | None = None
    """Custom LLM prompt (None = use built-in default)."""


@dataclasses.dataclass(frozen=True)
class ConvertResult:
    """Result of converting a PDF to Markdown."""

    markdown: str
    """Plain Markdown with ``<!--page-->`` markers between pages."""


async def convert(
    pdf_bytes: bytes,
    config: Config,
    *,
    markdown: str | None = None,
) -> ConvertResult:
    """Convert a PDF to plain Markdown via LLM.

    Splits the PDF into chunks, converts each chunk to Markdown concurrently
    via a vision-capable LLM, then joins the results with ``<!--page-->``
    separators.

    Args:
        pdf_bytes: Raw PDF file bytes.
        config: Conversion configuration.
        markdown: Optional pre-generated Markdown (skips LLM call).

    Returns:
        ConvertResult with the generated Markdown.
    """
    if markdown is not None:
        return ConvertResult(markdown=markdown)

    provider = PydanticAIMarkdownProvider(config.model, prompt=config.prompt)
    doc_chunks = list(chunks(pdf_bytes, page_count=config.page_count))

    markdown_chunks = list(await asyncio.gather(*[provider.generate_markdown(chunk) for chunk in doc_chunks]))

    numbered = renumber_markers(markdown_chunks)
    return ConvertResult(markdown="\n\n<!--page-->\n\n".join(numbered))
