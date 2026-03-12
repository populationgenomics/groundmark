import dataclasses

from anchorite import process_document
from anchorite.document import DocumentChunk, chunks

from groundmark.markdown import PydanticAIMarkdownProvider
from groundmark.parse import PdfplumberAnchorProvider


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for the groundmark processing pipeline."""

    model: str
    """Pydantic AI model string (e.g. "bedrock:au.anthropic.claude-sonnet-4-6")."""
    uniqueness_threshold: float = 0.5
    """Minimum score ratio between best and second-best alignment match."""
    min_overlap: float = 0.9
    """Minimum overlap fraction required for a valid alignment match."""
    page_count: int | None = None
    """Pages per chunk (None = whole PDF in one chunk)."""
    prompt: str | None = None
    """Custom LLM prompt (None = use built-in default)."""


@dataclasses.dataclass
class ProcessResult:
    """Result of processing a PDF into annotated Markdown."""

    annotated_markdown: str
    """Markdown with <span data-bbox="..." data-page="N"> tags."""
    coverage_percent: float
    """Fraction of markdown content covered by aligned bounding boxes."""


async def process(
    pdf_bytes: bytes,
    config: Config,
    *,
    markdown: str | None = None,
) -> ProcessResult:
    """Process a PDF into annotated Markdown with bounding box spans.

    Args:
        pdf_bytes: Raw PDF file bytes.
        config: Processing configuration.
        markdown: Optional pre-generated markdown (skips LLM call).

    Returns:
        ProcessResult with annotated markdown and coverage stats.
    """
    md_provider: _CachedMarkdownProvider | PydanticAIMarkdownProvider
    if markdown is not None:
        # Cached markdown corresponds to the whole PDF — don't chunk.
        md_provider = _CachedMarkdownProvider(markdown)
        doc_chunks = chunks(pdf_bytes)
    else:
        md_provider = PydanticAIMarkdownProvider(config.model, prompt=config.prompt)
        doc_chunks = chunks(pdf_bytes, page_count=config.page_count)

    result = await process_document(
        doc_chunks,
        markdown_provider=md_provider,
        anchor_provider=PdfplumberAnchorProvider(),
        alignment_uniqueness_threshold=config.uniqueness_threshold,
        alignment_min_overlap=config.min_overlap,
    )

    return ProcessResult(
        annotated_markdown=result.annotate(),
        coverage_percent=result.coverage_percent,
    )


class _CachedMarkdownProvider:
    """Returns pre-generated markdown, ignoring the chunk."""

    def __init__(self, markdown: str) -> None:
        self._markdown = markdown

    async def generate_markdown(self, _chunk: DocumentChunk) -> str:
        return self._markdown
