"""Core data types."""

import dataclasses


@dataclasses.dataclass(frozen=True, order=True)
class BBox:
    """A bounding box in normalized coordinates (0-1000 scale).

    Represents a rectangular region on a PDF page, with the origin at the
    top-left corner.
    """

    top: int
    """Top coordinate (y-min: [0-1000])."""
    left: int
    """Left coordinate (x-min: [0-1000])."""
    bottom: int
    """Bottom coordinate (y-max: [0-1000])."""
    right: int
    """Right coordinate (x-max: [0-1000])."""
