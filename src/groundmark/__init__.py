import warnings

from groundmark.resolve import DocumentIndex
from groundmark.types import BBox

__all__ = ["BBox", "DocumentIndex"]

# Convert functionality requires pydantic-ai (optional dependency).
# Import explicitly: from groundmark.convert import convert, Config, ModelConfig

warnings.warn(
    "groundmark is deprecated and its repository is being archived. "
    "Its citation-resolution API has been subsumed by anchorite "
    "(https://github.com/populationgenomics/anchorite) >= 0.4.0. "
    "Note that anchorite is not batteries-included: the PDF-to-Markdown / "
    "pydantic-ai layer is left to the caller; see "
    "https://github.com/populationgenomics/flowa for an example client.",
    DeprecationWarning,
    stacklevel=2,
)
