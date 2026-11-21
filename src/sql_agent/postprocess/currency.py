"""Currency converter: 'USD$ 12.30' -> number in the base currency. Never touches the DB."""

from typing import Dict, Optional

import pandas as pd

from src.sql_agent.postprocess.base import Converter
from utils.helpers import convert_price_to_sgd, detect_foreign_currencies, is_price_column


class CurrencyConverter(Converter):
    name = "currency"

    def __init__(self, cfg: Dict):
        self.base = cfg["base_currency"]
        # Rates are expressed relative to the base currency.
        self.rates: Dict[str, float] = cfg["conversion_rates"]
        self.label = f"Currency values (convert to {self.base})"

    def detect(self, series: pd.Series) -> bool:
        """A currency-like column that contains at least one non-base currency."""
        if not is_price_column(series):
            return False
        return detect_foreign_currencies(series.to_frame("c"), ["c"], self.base)

    def convert_value(self, value) -> Optional[float]:
        return convert_price_to_sgd(value, self.rates)

    def describe(self) -> str:
        return f"to {self.base}"
