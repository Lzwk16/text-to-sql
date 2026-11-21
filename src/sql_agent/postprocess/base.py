"""Converter interface for numeric post-processing of query results."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Applied:
    """One column that a converter actually changed."""

    column: str
    converter: str
    detail: str  # e.g. "to SGD"

    def as_dict(self) -> dict:
        return {"column": self.column, "converter": self.converter, "detail": self.detail}


class Converter:
    """Detects columns holding numbers-as-text and converts them to plain numbers."""

    name: str = ""
    label: str = ""

    def detect(self, series: pd.Series) -> bool:
        """True if this column needs this converter."""
        raise NotImplementedError

    def convert_value(self, value) -> Optional[float]:
        """Convert one value; None if it cannot be parsed."""
        raise NotImplementedError

    def describe(self) -> str:
        """Short phrase for the response and summary, e.g. 'to SGD'."""
        raise NotImplementedError

    def convert_series(self, series: pd.Series) -> pd.Series:
        return series.map(self.convert_value)
