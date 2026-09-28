"""Metadata describing what a mapped field means."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

FieldKind = Literal["continuous", "binary", "categorical"]


@dataclass(frozen=True)
class FieldSpec:
    """Semantic metadata for a predicted or observed grid field.

    Numeric values alone are ambiguous. A value of ``1`` can mean landslide,
    erosion, deposition, inundation or a severity class. FieldSpec carries the
    minimum metadata needed to compare two mapped fields without guessing.

    ``nodata`` is one exact value treated as missing. ``valid_range`` (low, high)
    marks everything outside it as missing too, and the count is reported: use it
    for flag values that drift, such as 9999.137 in a model output.

    ``sign_convention="positive_is_loss"`` marks a signed field where positive
    means loss (erosion depth); maps then keep loss red and gain blue.
    """

    name: str
    kind: FieldKind
    variable: str | None = None
    units: str | None = None
    positive_class: str | int | float | None = None
    threshold: float | None = None
    classes: Mapping[int | float | str, str] | None = None
    nodata: int | float | None = None
    valid_range: tuple[float, float] | None = None
    sign_convention: Literal["positive_is_gain", "negative_is_loss", "positive_is_loss"] | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("continuous", "binary", "categorical"):
            raise ValueError(
                "FieldSpec.kind must be 'continuous', 'binary' or 'categorical'"
            )
        if self.kind == "categorical" and self.classes is None:
            raise ValueError("categorical FieldSpec requires a classes mapping")
        if self.units == "probability" and self.kind != "continuous":
            raise ValueError("probability fields should use kind='continuous'")
        if self.valid_range is not None:
            low, high = self.valid_range
            if not low < high:
                raise ValueError(f"{self.name}: valid_range must be (low, high) with low < high")
