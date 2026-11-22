"""Currency converter: 'USD$ 12.30' -> number in the base currency. Never touches the DB."""

import re
from typing import Dict, List, Optional

import pandas as pd

from src.sql_agent.postprocess.base import Converter
from utils.helpers import convert_price_to_sgd

_CURRENCY = re.compile(r"^\s*([A-Z]{3})[\$€£]?\s*[\d,]+\.?\d*")


class CurrencyConverter(Converter):
    name = "currency"

    @classmethod
    def option_specs(cls, cfg: Dict) -> List[dict]:
        return [{
            "key": "base_currency", "label": "Convert currency to",
            "choices": list(cfg["conversion_rates"]), "default": cfg["base_currency"],
        }]

    @classmethod
    def apply_options(cls, cfg: Dict, chosen: Dict[str, str]) -> Dict:
        new = chosen.get("base_currency")
        if not new or new == cfg["base_currency"]:
            return cfg
        rates = cfg["conversion_rates"]
        # Rates are multipliers into the current base; re-express them in `new`.
        return {**cfg, "base_currency": new,
                "conversion_rates": {c: r / rates[new] for c, r in rates.items()}}

    def __init__(self, cfg: Dict):
        self.base = cfg["base_currency"]
        # Rates are expressed relative to the base currency.
        self.rates: Dict[str, float] = cfg["conversion_rates"]
        self.label = f"Currency values (convert to {self.base})"

    def _code(self, value) -> Optional[str]:
        """The currency code of a value like 'USD$ 12.30', only if it is a known one.

        Codes must be in the configured rates: 'DRV0130' (a driver id) looks like
        three capitals plus digits but is not a currency.
        """
        if not isinstance(value, str):
            return None
        m = _CURRENCY.match(value)
        return m.group(1) if m and m.group(1) in self.rates else None

    def matches(self, value: str) -> bool:
        return self._code(value) is not None

    def detect(self, series: pd.Series) -> bool:
        """A currency column (known codes) that has at least one non-base currency."""
        values = [v for v in series.dropna() if isinstance(v, str)]
        if not values:
            return False
        codes = [c for c in (self._code(v) for v in values) if c]
        if len(codes) / len(values) < self.min_match_ratio:
            return False
        return any(c != self.base for c in codes)

    def convert_value(self, value) -> Optional[float]:
        if self._code(value) is None:
            return None  # unknown code: NULL, never silently treated as the base
        return convert_price_to_sgd(value, self.rates)

    def describe(self, sample=None) -> str:
        return f"to {self.base}"
