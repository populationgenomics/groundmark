# groundmark

<img src="https://raw.githubusercontent.com/populationgenomics/groundmark/main/groundmark.webp" alt="groundmark" width="200">

PDF to Markdown conversion and quote-to-bbox resolution.

## What it does

1. **Convert**: Send PDF pages to a vision-capable LLM (via [Pydantic AI](https://ai.pydantic.dev/)) to produce clean Markdown with `<!--page-->` markers between pages.
2. **Resolve**: Given verbatim quote strings, locate them in the source PDF and return bounding box coordinates. Uses [pypdfium2](https://github.com/nickel-ern/pypdfium2) for per-character bbox extraction and [seq-smith](https://github.com/populationgenomics/seq-smith) for Smith-Waterman alignment.

## Quick Start

```python
import asyncio
from groundmark import DocumentIndex
from groundmark.convert import Config, ModelConfig, convert

async def main():
    pdf_bytes = open("document.pdf", "rb").read()

    # PDF -> Markdown (requires pydantic-ai, install with e.g. groundmark[bedrock])
    model = ModelConfig(name="bedrock:au.anthropic.claude-sonnet-4-6")
    result = await convert(pdf_bytes, Config(model=model))
    print(result.markdown[:500])

    # Resolve verbatim quotes to PDF bounding boxes
    doc = DocumentIndex(pdf_bytes)
    resolved = doc.resolve(["the patient presented with"])
    # -> {"the patient presented with": [(page, BBox(top, left, bottom, right)), ...]}

    # The DocumentIndex can be reused for multiple resolve calls against the same PDF
    more = doc.resolve(["another quote from the same paper"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Installation

```bash
# Resolve only (no LLM dependencies)
uv add groundmark

# With LLM provider extra(s) for conversion
uv add groundmark --extra anthropic,bedrock,google,openai
```

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
