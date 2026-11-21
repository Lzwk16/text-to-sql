"""Pluggable numeric converters applied to query results (never to the database)."""

from typing import Dict, Iterable, List, Tuple

import pandas as pd

from src.sql_agent.postprocess.base import Applied, Converter
from src.sql_agent.postprocess.currency import CurrencyConverter

# name -> converter class. Add new converters here.
REGISTRY: Dict[str, type] = {"currency": CurrencyConverter}

# Fixed application order. A column claimed by one converter is skipped by later ones.
ORDER: List[str] = ["currency"]


class UnknownConverter(ValueError):
    pass


def validate_names(names: Iterable[str]) -> List[str]:
    names = list(names)
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise UnknownConverter(
            f"Unknown conversion(s): {', '.join(unknown)}. "
            f"Available: {', '.join(REGISTRY)}."
        )
    return names


def build_converters(postprocess_cfg: dict, enabled: Iterable[str]) -> List[Converter]:
    """Instantiate the enabled converters in fixed order."""
    wanted = set(validate_names(enabled))
    return [
        REGISTRY[name](postprocess_cfg[name])
        for name in ORDER
        if name in wanted and name in postprocess_cfg
    ]


def available(postprocess_cfg: dict) -> List[dict]:
    """Converters that are configured, for the UI: [{name, label}]."""
    out = []
    for name in ORDER:
        if name in postprocess_cfg:
            out.append(
                {"name": name, "label": REGISTRY[name](postprocess_cfg[name]).label}
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
            out[col] = conv.convert_series(out[col])
            claimed.add(col)
            applied.append(Applied(str(col), conv.name, conv.describe()))
    return out, applied
