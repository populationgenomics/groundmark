# groundmark

<img src="https://raw.githubusercontent.com/populationgenomics/groundmark/main/groundmark.webp" alt="groundmark" width="200">

## Grounded Markdown for PDFs

**groundmark is a thin, batteries-included wrapper around [anchorite](https://github.com/populationgenomics/anchorite).** It provides concrete implementations of anchorite's provider protocols — [Pydantic AI](https://ai.pydantic.dev/) for LLM-based Markdown generation and [pdfplumber](https://github.com/jsvine/pdfplumber) for bounding box extraction — so you can go from PDF bytes to annotated Markdown in a single call. All the heavy lifting (Smith-Waterman alignment, annotation, stripping, quote resolution) lives in anchorite.

Give it a PDF and a model string, get back Markdown with embedded bounding box coordinates that trace every text span back to its location in the source PDF.

## Architecture

The library processes documents in two streams that are then merged:

1. **Semantic Stream**: The PDF is sent to an LLM (via Pydantic AI) to produce clean Markdown with `<!--page-->` markers between pages.
2. **Positional Stream**: The PDF is parsed locally by pdfplumber to extract line-level text segments and their bounding boxes.
3. **Alignment**: Smith-Waterman alignment (via anchorite) maps each parsed line to its position in the Markdown, constrained by page boundaries.
4. **Annotation**: Bounding box coordinates are injected as HTML span attributes:

   ```html
   <span data-bbox="120,45,180,890" data-page="3">The patient presented with</span>
   ```

## Quick Start

```python
import asyncio
import groundmark as gm

async def main():
    pdf_bytes = open("document.pdf", "rb").read()

    config = gm.Config(model="bedrock:au.anthropic.claude-sonnet-4-6")

    # PDF -> annotated Markdown (one call)
    result = await gm.process(pdf_bytes, config)
    print(f"Coverage: {result.coverage_percent:.2%}")
    print(result.annotated_markdown[:500])

    # Strip for LLM consumption
    stripped = gm.strip(result.annotated_markdown)
    # stripped.plain_text: clean Markdown with spans removed
    # stripped.validation_map: list of (start, end, Anchor) ranges

    # Resolve verbatim quotes to PDF coordinates
    resolved = gm.resolve(result.annotated_markdown, ["the patient presented with"])
    # -> {"the patient presented with": [(page, BBox), ...]}

if __name__ == "__main__":
    asyncio.run(main())
```

## Debug Visualizer

The included visualizer overlays extracted bounding boxes onto the source PDF, useful for diagnosing alignment issues. Blue highlights show raw extracted boxes from pdfplumber; red highlights show aligned boxes from the annotated Markdown.

```bash
python -m groundmark.visualize input.pdf output.pdf --model "bedrock:au.anthropic.claude-sonnet-4-6"

# Or with cached Markdown:
python -m groundmark.visualize input.pdf output.pdf --markdown cached.md
```

![Visualizer output showing blue (raw) and red (aligned) bounding box overlays](https://raw.githubusercontent.com/populationgenomics/groundmark/main/visualize_example.jpg)

*Screenshot from Santoro et al., "Health outcomes and drug utilisation in children with Noonan syndrome: a European cohort study," Orphanet J Rare Dis 20:76 (2025). [doi:10.1186/s13023-025-03594-7](https://doi.org/10.1186/s13023-025-03594-7). CC-BY 4.0.*

## Configuration

### Timeouts

The LLM call for PDF-to-Markdown conversion can take several minutes for large documents, especially with Opus on Bedrock. Timeout defaults by provider:

| Provider | Default | Environment Variable |
|----------|---------|---------------------|
| Bedrock (boto3) | 300s | `AWS_READ_TIMEOUT` |
| Anthropic (httpx) | 600s | — (use `ModelSettings(timeout=...)`) |

For Bedrock with Opus, 300s may not be enough. Set a higher timeout:

```bash
export AWS_READ_TIMEOUT=600
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
