"""Converter interface for numeric post-processing of query results."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


@dataclass
class Applied:
    """One column that a converter actually changed."""

    column: str
    converter: str
    detail: str  # e.g. "to SGD"

    def as_dict(self) -> dict:
        return {"column": self.column, "converter": self.converter, "detail": self.detail}


@dataclass
class Detected:
    """A database column that a converter recognised from sampled values."""

    table: str
    column: str
    converter: str
    example: Optional[str] = None  # a raw value, shown to the model only if allowed

    def as_dict(self) -> dict:
        return {"table": self.table, "column": self.column, "converter": self.converter}


class Converter:
    """Detects columns holding numbers-as-text and converts them to plain numbers."""

    name: str = ""
    label: str = ""
    kind: str = "number"  # "number" or "text" (e.g. ISO datetimes)

    @property
    def cache_key(self) -> str:
        """Part of the per-database detection cache key; override if detection
        depends on an option (the datetime day/month order does)."""
        return self.name

    @classmethod
    def option_specs(cls, cfg: dict) -> List[dict]:
        """User-selectable settings: [{key, label, choices, default}]."""
        return []

    @classmethod
    def apply_options(cls, cfg: dict, chosen: Dict[str, str]) -> dict:
        """Return ``cfg`` with the chosen option values applied (cfg is not mutated)."""
        return cfg

    @property
    def sql_name(self) -> str:
        """Name of the SQLite function registered for this converter."""
        return f"convert_{self.name}"

    min_match_ratio = 0.9  # share of sampled values that must match to claim a column

    def matches(self, value: str) -> bool:
        """True if one text value is in this converter's format (used by detect)."""
        raise NotImplementedError

    def detect(self, series: pd.Series) -> bool:
        """True if this column needs this converter.

        Default rule: at least ``min_match_ratio`` of the non-null text values match
        the format. Converters whose formats always need conversion use this as is.
        """
        values = [v for v in series.dropna() if isinstance(v, str)]
        if not values:
            return False
        hits = sum(1 for v in values if self.matches(v))
        return hits / len(values) >= self.min_match_ratio

    def convert_value(self, value) -> Optional[Any]:
        """Convert one value; None if it cannot be parsed."""
        raise NotImplementedError

    def describe(self, sample: Optional[Any] = None) -> str:
        """Short phrase for the response and summary, e.g. 'to SGD'.

        ``sample`` is one raw value of the column, for converters whose target
        depends on the value (units).
        """
        raise NotImplementedError

    def convert_series(self, series: pd.Series) -> pd.Series:
        return series.map(self.convert_value)

    def hint(self, table: str, column: str, example: Optional[str]) -> str:
        """One line telling the SQL-writing model how to use this converter."""
        eg = f" such as '{example}'" if example else ""
        return (
            f"`{table}.{column}` holds text{eg}. Use `{self.sql_name}({column})` "
            f"(a value {self.describe(example)}) in MIN, MAX, SUM, AVG, comparisons "
            f"and ORDER BY. Never apply those to the raw column."
        )

    def sql_function(self) -> Callable[[Any], Optional[Any]]:
        """Pure function for ``connection.create_function``; never raises."""

        def fn(value):
            try:
                return self.convert_value(value)
            except Exception:  # bad input becomes NULL, not a query failure
                return None

        return fn
