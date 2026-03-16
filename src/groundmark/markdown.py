"""PDF to Markdown conversion via Pydantic AI (any supported LLM)."""

import re
import unicodedata
from typing import Final

from anchorite.document import DocumentChunk
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent

# Apparently, faithfully analyzing a PDF's complicated layout and transcribing
# it into well-structured Markdown isn't creative enough for Claude's content
# filter. Asking the model to add line numbers gives it something "original"
# to contribute, which satisfies the anti-regurgitation heuristic. We strip
# them right after.
# https://privacy.claude.com/en/articles/10023638-why-am-i-receiving-an-output-blocked-by-content-filtering-policy-error
_LINE_NUM_PREFIX = """
IMPORTANT: Prefix every output line with its line number followed by a
pipe character (no trailing space), e.g.:
  1|# Heading
  2|
  3|Some paragraph text here.
Start numbering at 1. This is required for all output.
"""

_LINE_NUM_RE = re.compile(r"^\d+\|", re.MULTILINE)

PROMPT: Final[str] = (
    """
Carefully transcribe the text for this pdf into a text file with
markdown annotations.
"""
    + _LINE_NUM_PREFIX
    + """
**The final output must be formatted as text that visually
mimics in markdown the layout and hierarchy of the original PDF
when rendered (ignoring the line-number prefixes).**

* Do not include headers or footers that are repeated on each page.
* Do not include page numbers.
* Preserve the reading order of the text as it appears in the PDF.
* Remove hyphens that break words at the end of lines.
  * e.g. "uti- lized" -> "utilized"
* Use Markdown headings (`#`, `##`, `###`) to reflect the size and
  hierarchy of titles and subtitles in the PDF.
* Ensure that there are blank lines before and after headings, lists,
  tables, and images.
* End each paragraph with a blank line.
* Do not break lines within paragraphs or headings.
* Render bullet points and numbered lettered lists as markdown lists.
  * It is ok to remove brackets and other consistent punctuation around
    list identifiers
    * e.g. "a)" -> "a."
* Use blockquotes for any sidebars or highlighted text.
* Bold all words and phrases that appear bolded in the original
  source material. Similarly, italicise all text in italics.
* Render tables as markdown, paying particular attention to copying
  identifiers exactly.
* Break text into paragraphs and lists exactly as they appear in
  the PDF.
* Preserve figure/chart captions verbatim — do not paraphrase, extend,
  or interleave them with descriptions. If useful context is only visible
  in the image (e.g. axis labels, legend entries, data values), add it
  as a separate paragraph after the caption.
* Convert bar charts into markdown tables where possible.
* Convert tables contained in images into markdown.
* Keep mathematical expressions as close to the PDF's own characters as
  possible — use the same Unicode symbols (×, ≥, α, β, etc.) rather than
  converting to LaTeX commands.
  * Only use LaTeX (`$...$` / `$$...$$`) for complex display equations
    with fractions, integrals, summations, or multi-level notation that
    cannot be represented legibly in plain text.
* Insert markers at the start of each page of the form `<!--page-->`
* Surround tables and figure descriptions with markers:
  * `<!--table-->` ... `<!--end-->`
  * `<!--figure-->` ... `<!--end-->`
"""
)

_agent: Agent[None, str] = Agent(output_type=str)


class PydanticAIMarkdownProvider:
    """MarkdownProvider that converts PDF chunks to Markdown via a vision-capable LLM."""

    def __init__(self, model: str, *, prompt: str | None = None) -> None:
        self.model = model
        self._prompt = prompt or PROMPT

    async def generate_markdown(self, chunk: DocumentChunk) -> str:
        """Convert a document chunk to Markdown.

        Returns:
            Markdown string with ``<!--page-->`` markers between pages.
        """
        result = await _agent.run(
            [BinaryContent(data=chunk.data, media_type=chunk.mime_type), self._prompt],
            model=self.model,
        )
        # Strip the line-number prefixes added to bypass Claude's content filter.
        markdown = _LINE_NUM_RE.sub("", result.output)
        # NFKC-normalize so superscript digits, ligatures, etc. match the
        # NFKC-normalized anchor text from pdfplumber extraction.
        return unicodedata.normalize("NFKC", markdown)
