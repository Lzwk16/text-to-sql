"""Pluggable numeric converters applied to query results (never to the database)."""

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.sql_agent.postprocess.base import Applied, Converter
from src.sql_agent.postprocess.currency import CurrencyConverter
from src.sql_agent.postprocess.datetime_iso import DatetimeConverter
from src.sql_agent.postprocess.durations import DurationsConverter
from src.sql_agent.postprocess.numbers import NumbersConverter
from src.sql_agent.postprocess.units import UnitsConverter

# name -> converter class. Add new converters here.
REGISTRY: Dict[str, type] = {
    "currency": CurrencyConverter,
    "units": UnitsConverter,
    "durations": DurationsConverter,
    "numbers": NumbersConverter,
    "datetime": DatetimeConverter,
}

# Fixed application order. A column claimed by one converter is skipped by later ones.
ORDER: List[str] = ["currency", "units", "durations", "numbers", "datetime"]


AUTO = "auto"  # pseudo-name: enable every converter that detects a column in the DB


class UnknownConverter(ValueError):
    pass


def validate_names(names: Iterable[str]) -> List[str]:
    names = list(names)
    unknown = [n for n in names if n not in REGISTRY and n != AUTO]
    if unknown:
        raise UnknownConverter(
            f"Unknown conversion(s): {', '.join(unknown)}. "
            f"Available: {', '.join([AUTO, *REGISTRY])}."
        )
    return names


def validate_options(postprocess_cfg: dict, options: Optional[dict]) -> dict:
    """Check per-request converter options; raise ``UnknownConverter`` if invalid."""
    options = options or {}
    for name, chosen in options.items():
        if name not in REGISTRY or name not in postprocess_cfg:
            raise UnknownConverter(f"Unknown conversion '{name}' in conversion_options.")
        specs = {s["key"]: s for s in REGISTRY[name].option_specs(postprocess_cfg[name])}
        for key, value in (chosen or {}).items():
            if key not in specs:
                raise UnknownConverter(f"Unknown option '{name}.{key}'.")
            if value not in specs[key]["choices"]:
                raise UnknownConverter(
                    f"Invalid value '{value}' for {name}.{key}. "
                    f"Choose one of: {', '.join(specs[key]['choices'])}."
                )
    return options


def _configured(postprocess_cfg: dict, name: str, options: Optional[dict]) -> dict:
    cfg = postprocess_cfg[name]
    chosen = (options or {}).get(name)
    return REGISTRY[name].apply_options(cfg, chosen) if chosen else cfg


def build_converters(
    postprocess_cfg: dict, enabled: Iterable[str], options: Optional[dict] = None
) -> List[Converter]:
    """Instantiate the enabled converters in fixed order, with per-request options."""
    wanted = set(validate_names(enabled))
    return [
        REGISTRY[name](_configured(postprocess_cfg, name, options))
        for name in ORDER
        if name in wanted and name in postprocess_cfg
    ]


def build_all(postprocess_cfg: dict, options: Optional[dict] = None) -> List[Converter]:
    """Every configured converter, in fixed order (used for detection)."""
    return build_converters(
        postprocess_cfg, [n for n in ORDER if n in postprocess_cfg], options
    )


def resolve_names(requested: Iterable[str], detected_converters: Iterable[str]) -> List[str]:
    """Expand ``auto`` to the converters that detected something in this database."""
    names = [n for n in requested if n != AUTO]
    if AUTO in requested:
        names += [n for n in detected_converters if n not in names]
    return names


def available(postprocess_cfg: dict) -> List[dict]:
    """Converters that are configured, for the UI: [{name, label, options}]."""
    out = []
    if any(n in postprocess_cfg for n in ORDER):
        out.append({"name": AUTO, "label": "Auto-detect from this database", "options": []})
    for name in ORDER:
        if name in postprocess_cfg:
            out.append(
                {
                    "name": name,
                    "label": REGISTRY[name](postprocess_cfg[name]).label,
                    "options": REGISTRY[name].option_specs(postprocess_cfg[name]),
                }
            )
    return out


def apply_converters(
    df: pd.DataFrame, converters: List[Converter]
) -> Tuple[pd.DataFrame, List[Applied]]:
    """Convert detected columns. Returns a new frame; the input is not modified."""
    out = df.copy()
    applied: List[Applied] = []
    claimed = set()
    for conv in converters:
        for col in out.columns:
            if col in claimed or not conv.detect(out[col]):
                continue
            sample = next(iter(out[col].dropna()), None)  # raw value, before converting
            out[col] = conv.convert_series(out[col])
            claimed.add(col)
            applied.append(Applied(str(col), conv.name, conv.describe(sample)))
    return out, applied
