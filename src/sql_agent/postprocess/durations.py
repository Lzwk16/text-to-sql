"""Durations: '2h 30m', '1d 4h', '90 min', '1:30:00' -> one target unit."""

import re
from typing import Dict, List, Optional

from src.sql_agent.postprocess.base import Converter

_SECONDS = {"d": 86400, "h": 3600, "m": 60, "s": 1}
# token -> unit letter. A bare 'm' means minutes only inside a compound duration;
# a single '5 m' is metres and belongs to the units converter.
_TOKENS = {
    "d": "d", "day": "d", "days": "d",
    "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
    "m": "m", "min": "m", "mins": "m", "minute": "m", "minutes": "m",
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
}
_PART = re.compile(r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)")
_CLOCK = re.compile(r"^(\d{1,3}):([0-5]\d):([0-5]\d)$")  # H:MM:SS (MM:SS is a clock time)
_TARGETS = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}


class DurationsConverter(Converter):
    name = "durations"
    label = "Durations (2h 30m, 1:30:00)"

    @classmethod
    def option_specs(cls, cfg: Optional[Dict]) -> List[dict]:
        return [{"key": "target", "label": "Convert durations to",
                 "choices": list(_TARGETS), "default": (cfg or {}).get("target", "minutes")}]

    @classmethod
    def apply_options(cls, cfg: Optional[Dict], chosen: Dict[str, str]) -> Dict:
        return {**(cfg or {}), **chosen}

    def __init__(self, cfg: Optional[Dict] = None):
        self.target = (cfg or {}).get("target", "minutes")
        if self.target not in _TARGETS:
            raise ValueError(f"durations.target must be one of {sorted(_TARGETS)}")

    def _seconds(self, value: str) -> Optional[float]:
        v = value.strip()
        clock = _CLOCK.match(v)
        if clock:
            h, m, s = (int(x) for x in clock.groups())
            return h * 3600 + m * 60 + s
        parts = _PART.findall(v)
        if not parts or _PART.sub("", v).strip():  # leftovers: not a duration
            return None
        units = []
        for _, tok in parts:
            unit = _TOKENS.get(tok.lower())
            if unit is None:
                return None
            units.append((tok.lower(), unit))
        if len(parts) == 1 and units[0][0] == "m":
            return None  # a lone 'm' is metres (units converter)
        if len({u for _, u in units}) != len(units):
            return None  # '5h 3h' is not a duration
        return sum(float(n) * _SECONDS[u] for (n, _), (_, u) in zip(parts, units))

    def matches(self, value: str) -> bool:
        return self._seconds(value) is not None

    def convert_value(self, value) -> Optional[float]:
        if not isinstance(value, str):
            return None
        seconds = self._seconds(value)
        return None if seconds is None else round(seconds / _TARGETS[self.target], 6)

    def describe(self, sample=None) -> str:
        return f"to {self.target}"
