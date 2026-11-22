"""Dates and datetimes in common text formats -> ISO 8601 text.

'05/10/1973' -> '1973-10-05' (day first by default), '01/01/2023 0:00' ->
'2023-01-01T00:00:00'. A value is converted only when exactly one reading is
reliable: a column that mixes day-first and month-first dates is left alone.
"""

import re
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.sql_agent.postprocess.base import Converter

# Month-name formats go through strptime (rare, so speed does not matter).
_NAMED = ["%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y"]
# Numeric formats are read with regexes: this runs once per row inside SQLite, and
# trying a dozen strptime formats per value was ~66 us (a 133k-row MIN timed out).
_TIME = r"(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?"
_YEAR_FIRST = re.compile(rf"^(\d{{4}})([-/])(\d{{1,2}})\2(\d{{1,2}}){_TIME}$")
_NUMERIC = re.compile(rf"^(\d{{1,2}})([-/])(\d{{1,2}})\2(\d{{4}}){_TIME}$")


def _build(y, mo, d, h=None, mi=None, s=None) -> Optional[datetime]:
    try:
        return datetime(int(y), int(mo), int(d), int(h or 0), int(mi or 0), int(s or 0))
    except ValueError:
        return None


@lru_cache(maxsize=1 << 16)
def _named(value: str) -> Optional[datetime]:
    for fmt in _NAMED:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _iso(dt: datetime, has_time: bool) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S") if has_time else dt.strftime("%Y-%m-%d")


def _has_time(value: str) -> bool:
    return ":" in value


class DatetimeConverter(Converter):
    name = "datetime"
    label = "Dates and times (convert to ISO 8601)"
    kind = "text"

    _ORDERS = {"day first": True, "month first": False}

    @classmethod
    def option_specs(cls, cfg: Optional[Dict]) -> List[dict]:
        dayfirst = bool((cfg or {}).get("dayfirst", True))
        return [{"key": "order", "label": "Read 05/10/2023 as",
                 "choices": list(cls._ORDERS),
                 "default": "day first" if dayfirst else "month first"}]

    @classmethod
    def apply_options(cls, cfg: Optional[Dict], chosen: Dict[str, str]) -> Dict:
        out = dict(cfg or {})
        if "order" in chosen:
            out["dayfirst"] = cls._ORDERS[chosen["order"]]
        return out

    @property
    def cache_key(self) -> str:
        return f"datetime:{self.dayfirst}"  # detection depends on the order

    def __init__(self, cfg: Optional[Dict] = None):
        self.dayfirst = bool((cfg or {}).get("dayfirst", True))

    def _reading(self, value: str) -> Tuple[Optional[datetime], bool]:
        """(parsed, contradicts_config). Contradicts: only the other order fits."""
        v = value.strip()
        m = _YEAR_FIRST.match(v)
        if m:
            y, _, mo, d, h, mi, sec = m.groups()
            return _build(y, mo, d, h, mi, sec), False
        m = _NUMERIC.match(v)
        if m:
            a, _, b, y, h, mi, sec = m.groups()
            day, month = (a, b) if self.dayfirst else (b, a)
            dt = _build(y, month, day, h, mi, sec)
            if dt is not None:
                return dt, False
            dt = _build(y, day, month, h, mi, sec)  # the opposite reading
            return (dt, True) if dt is not None else (None, False)
        return _named(v), False

    def matches(self, value: str) -> bool:
        dt, contradicts = self._reading(value)
        return dt is not None and not contradicts

    def detect(self, series: pd.Series) -> bool:
        values = [v.strip() for v in series.dropna() if isinstance(v, str)]
        if not values:
            return False
        parsed = [self._reading(v) for v in values]
        if any(contradicts for _, contradicts in parsed):
            return False  # mixed or opposite day/month order: do not guess
        ok = [(dt, v) for (dt, _), v in zip(parsed, values) if dt is not None]
        if len(ok) / len(values) < self.min_match_ratio:
            return False
        # Needs converting only if some value is not already in ISO form.
        return any(_iso(dt, _has_time(v)) != v for dt, v in ok)

    def convert_value(self, value) -> Optional[str]:
        if not isinstance(value, str):
            return None
        dt, contradicts = self._reading(value)
        if dt is None or contradicts:
            return None
        return _iso(dt, _has_time(value))

    def describe(self, sample=None) -> str:
        order = "day first" if self.dayfirst else "month first"
        return f"to ISO 8601 ({order} when ambiguous)"
