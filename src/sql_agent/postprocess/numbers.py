"""Formatted numbers: '1,234.50', '1.2K', '3M', '(123)' -> plain numbers."""

import re
from typing import Dict, Optional

_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}  # upper case only: 'm' is a unit
_NUM = r"\d+(?:\.\d+)?"
_GROUPED = re.compile(rf"^\(?-?\$?\s?\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?\)?$")
_PAREN = re.compile(rf"^\(\s?\$?\s?{_NUM}\)$")  # accounting negative: (123)
_SUFFIXED = re.compile(rf"^\(?-?\$?\s?({_NUM})\s?([KMBT])\)?$")

from src.sql_agent.postprocess.base import Converter  # noqa: E402


class NumbersConverter(Converter):
    """Plain digits like '12345' are deliberately not matched (nothing to convert)."""

    name = "numbers"
    label = "Formatted numbers (1,234.50 / 1.2K / (123))"

    def __init__(self, cfg: Optional[Dict] = None):
        pass

    def matches(self, value: str) -> bool:
        v = value.strip()
        return bool(_GROUPED.match(v) or _SUFFIXED.match(v) or _PAREN.match(v))

    def convert_value(self, value) -> Optional[float]:
        if not isinstance(value, str) or not self.matches(value):
            return None
        v = value.strip()
        negative = v.startswith("(") or v.startswith("-") or v.startswith("(-")
        mult = 1.0
        m = _SUFFIXED.match(v)
        if m:
            mult = _SUFFIX[m.group(2)]
            digits = m.group(1)
        else:
            digits = re.sub(r"[^\d.]", "", v)
        number = float(digits) * mult
        return round(-number if negative else number, 6)

    def describe(self, sample=None) -> str:
        return "to plain numbers"
