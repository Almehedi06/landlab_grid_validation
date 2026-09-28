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
    """

    name: str
    kind: FieldKind
    variable: str | None = None
    units: str | None = None
    positive_class: str | int | float | None = None
    threshold: float | None = None
    classes: Mapping[int | float | str, str] | None = None
    nodata: int | float | None = None
    sign_convention: Literal["positive_is_gain", "negative_is_loss"] | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("continuous", "binary", "categorical"):
            raise ValueError(
                "FieldSpec.kind must be 'continuous', 'binary' or 'categorical'"
            )
        if self.kind == "categorical" and self.classes is None:
            raise ValueError("categorical FieldSpec requires a classes mapping")
        if self.units == "probability" and self.kind != "continuous":
            raise ValueError("probability fields should use kind='continuous'")

