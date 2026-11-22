"""Units: '5 kg', '12.5 lb', '30 cm', '98 °F' -> one target unit per dimension.

pint resolves each unit token to a conversion factor once (cached), so converting
100k rows costs a regex and a dict lookup per row, not a pint parse.
"""

import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pint

from src.sql_agent.postprocess.base import Converter

_ureg = pint.UnitRegistry()  # one shared registry (pint cannot mix registries)
_Q = _ureg.Quantity

# token (lower case) -> (pint unit, dimension). A conservative whitelist: ambiguous
# tokens are decided here. 'oz' is mass, 'gal' is the US gallon, bare 'c'/'f'/'k'
# are not accepted, and time units belong to the durations converter.
_TOKENS: Dict[str, Tuple[str, str]] = {
    **{t: (u, "length") for t, u in {
        "mm": "millimeter", "cm": "centimeter", "m": "meter", "km": "kilometer",
        "in": "inch", "ft": "foot", "yd": "yard", "mi": "mile",
        "meter": "meter", "meters": "meter", "metre": "meter", "metres": "meter",
        "kilometer": "kilometer", "kilometers": "kilometer",
        "kilometre": "kilometer", "kilometres": "kilometer",
        "mile": "mile", "miles": "mile", "foot": "foot", "feet": "foot",
        "inch": "inch", "inches": "inch",
    }.items()},
    **{t: (u, "mass") for t, u in {
        "mg": "milligram", "g": "gram", "kg": "kilogram", "lb": "pound",
        "lbs": "pound", "oz": "ounce", "gram": "gram", "grams": "gram",
        "kilogram": "kilogram", "kilograms": "kilogram", "pound": "pound",
        "pounds": "pound", "ounce": "ounce", "ounces": "ounce",
    }.items()},
    **{t: (u, "volume") for t, u in {
        "ml": "milliliter", "l": "liter", "litre": "liter", "litres": "liter",
        "liter": "liter", "liters": "liter", "gal": "gallon", "gallon": "gallon",
        "gallons": "gallon",
    }.items()},
    **{t: (u, "temperature") for t, u in {
        "°c": "degC", "° c": "degC", "degc": "degC", "celsius": "degC",
        "°f": "degF", "° f": "degF", "degf": "degF", "fahrenheit": "degF",
    }.items()},
}
_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(° ?[A-Za-z]|[A-Za-z]+)\s*$")
_DEFAULT_TARGETS = {"length": "km", "mass": "kg", "volume": "l", "temperature": "C"}
_TEMP_UNIT = {"C": "degC", "F": "degF"}
_DIMENSIONALITY = {
    "length": "[length]", "mass": "[mass]", "volume": "[length] ** 3",
    "temperature": "[temperature]",
}


@lru_cache(maxsize=None)
def _affine(unit: str, target: str) -> Tuple[float, float]:
    """(offset, scale) with result = offset + scale * value, via pint at two points.

    Linear for ordinary units; carries the offset for temperature. Built from
    ``Q_(v, unit)`` because ``v * ureg.degC`` raises OffsetUnitCalculusError.
    """
    at0 = _Q(0.0, unit).to(target).magnitude
    at1 = _Q(1.0, unit).to(target).magnitude
    return at0, at1 - at0


class UnitsConverter(Converter):
    name = "units"
    label = "Units (kg, cm, °F, ... converted to your targets)"

    _CHOICES = {
        "length": ["mm", "cm", "m", "km", "in", "ft", "yd", "mi"],
        "mass": ["mg", "g", "kg", "lb", "oz"],
        "volume": ["ml", "l", "gal"],
        "temperature": ["C", "F"],
    }

    @classmethod
    def option_specs(cls, cfg: Optional[Dict]) -> List[dict]:
        targets = {**_DEFAULT_TARGETS, **(((cfg or {}).get("targets")) or {})}
        out = []
        for dim, choices in cls._CHOICES.items():
            default = targets[dim]
            out.append({
                "key": dim, "label": f"{dim.capitalize()} unit",
                "choices": choices if default in choices else [default, *choices],
                "default": default,
            })
        return out

    @classmethod
    def apply_options(cls, cfg: Optional[Dict], chosen: Dict[str, str]) -> Dict:
        cfg = cfg or {}
        return {**cfg, "targets": {**(cfg.get("targets") or {}), **chosen}}

    def __init__(self, cfg: Optional[Dict] = None):
        targets = {**_DEFAULT_TARGETS, **((cfg or {}).get("targets") or {})}
        self.targets: Dict[str, str] = {}
        for dim, target in targets.items():
            if dim not in _DIMENSIONALITY:
                raise ValueError(f"Unknown unit dimension '{dim}' in units.targets")
            pint_target = _TEMP_UNIT.get(target, target) if dim == "temperature" else target
            try:
                dimensionality = str(_Q(1, pint_target).dimensionality)
            except Exception as exc:  # pint.UndefinedUnitError etc.
                raise ValueError(f"units.targets.{dim}: unknown unit '{target}'") from exc
            if dimensionality != _DIMENSIONALITY[dim]:
                raise ValueError(f"units.targets.{dim}: '{target}' is not a {dim} unit")
            self.targets[dim] = pint_target
        self._labels = {d: t for d, t in targets.items()}

    def _parse(self, value: str) -> Optional[Tuple[float, str, str]]:
        m = _PATTERN.match(value)
        if not m:
            return None
        entry = _TOKENS.get(m.group(2).lower())
        if entry is None:
            return None
        unit, dim = entry
        return float(m.group(1)), unit, dim

    def matches(self, value: str) -> bool:
        return self._parse(value) is not None

    def detect(self, series) -> bool:
        """Claim a column only if its values share one dimension (kg with cm is not)."""
        values = [v for v in series.dropna() if isinstance(v, str)]
        if not values:
            return False
        parsed = [self._parse(v) for v in values]
        dims = [p[2] for p in parsed if p]
        if len(dims) / len(values) < self.min_match_ratio:
            return False
        return max(dims.count(d) for d in set(dims)) / len(dims) >= self.min_match_ratio

    def convert_value(self, value) -> Optional[float]:
        if not isinstance(value, str):
            return None
        parsed = self._parse(value)
        if parsed is None:
            return None
        number, unit, dim = parsed
        offset, scale = _affine(unit, self.targets[dim])
        return round(offset + scale * number, 6)

    def describe(self, sample=None) -> str:
        if isinstance(sample, str):
            parsed = self._parse(sample)
            if parsed:
                dim = parsed[2]
                return f"to {self._labels[dim]} ({dim})"
        return "to " + ", ".join(self._labels.values())
