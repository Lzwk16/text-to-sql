"""
Utility functions for text-to-SQL application.

This module provides helper functions for:
- Currency conversion to SGD
- Price column detection
- SQL output cleaning
"""

import re
from typing import Dict

import pandas as pd


def convert_price_to_sgd(price_str, conversion_rates: Dict[str, float]) -> float:
    """
    Convert price string to SGD float.

    Args:
        price_str: Price string in format 'SGD$ 123.45' or 'USD$ 123.45'
        conversion_rates: Dictionary mapping currency codes to SGD rates

    Returns:
        Price converted to SGD as float, or None if invalid

    Examples:
        >>> convert_price_to_sgd('SGD$ 100.00', {'SGD': 1.0, 'USD': 1.35})
        100.0
        >>> convert_price_to_sgd('USD$ 100.00', {'SGD': 1.0, 'USD': 1.35})
        135.0
    """
    if pd.isna(price_str) or price_str is None:
        return None

    match = re.match(r"([A-Z]{3})[\$€£]?\s*([\d,]+\.?\d*)", str(price_str))

    if match:
        currency = match.group(1)
        amount = float(match.group(2).replace(",", ""))
        rate = conversion_rates.get(currency, 1.0)
        return round(amount * rate, 2)

    return None


def is_price_column(series: pd.Series) -> bool:
    """
    Check if a pandas series contains price data.

    Args:
        series: Pandas series to check

    Returns:
        True if series contains price data (currency format strings)

    Examples:
        >>> s = pd.Series(['SGD$ 100.00', 'USD$ 200.00'])
        >>> is_price_column(s)
        True
        >>> s = pd.Series(['apple', 'banana'])
        >>> is_price_column(s)
        False
    """
    first_val = series.dropna().iloc[0] if not series.dropna().empty else None

    if first_val is None:
        return False

    if isinstance(first_val, str):
        return bool(re.match(r"([A-Z]{3})[\$€£]?\s*([\d,]+\.?\d*)", str(first_val)))

    return False


def detect_foreign_currencies(
    df: pd.DataFrame, price_columns: list, base_currency: str = "SGD"
) -> bool:
    """
    Detect if any price columns contain foreign (non-base) currencies.

    Args:
        df: DataFrame to check
        price_columns: List of price column names to check
        base_currency: Base currency code (default: 'SGD')

    Returns:
        True if foreign currencies detected

    Examples:
        >>> df = pd.DataFrame({'price': ['SGD$ 100', 'USD$ 200']})
        >>> detect_foreign_currencies(df, ['price'])
        True
        >>> df = pd.DataFrame({'price': ['SGD$ 100', 'SGD$ 200']})
        >>> detect_foreign_currencies(df, ['price'])
        False
    """
    for col in price_columns:
        if col in df.columns:
            for val in df[col].dropna().head(100):
                if isinstance(val, str):
                    match = re.match(r"([A-Z]{3})", str(val))
                    if match and match.group(1) != base_currency:
                        return True
    return False


def clean_sql_output(sql_text: str) -> str:
    """
    Clean SQL output from LLM by removing special tokens and artifacts.

    Removes:
    - Special tokens: <s>, </s>, <pad>, <unk>, etc.
    - Thinking tags: <think>...</think>
    - SQL markers: [SQL], [/SQL]
    - Extra whitespace and newlines

    Args:
        sql_text: Raw SQL output from LLM

    Returns:
        Cleaned SQL query string

    Examples:
        >>> clean_sql_output('<s>SELECT * FROM table;</s>')
        'SELECT * FROM table;'
        >>> clean_sql_output('SELECT * FROM table;[/SQL] extra text')
        'SELECT * FROM table;'
    """
    # Remove special tokens using simple string replacement (no regex)
    special_tokens = [
        "<s>",
        "</s>",  # Start/end of sequence
        "<pad>",
        "<unk>",  # Padding and unknown tokens
        "[SQL]",
        "[/SQL]",  # SQL markers
        "<think>",
        "</think>",  # Thinking tags
        "<|endoftext|>",  # GPT-style end token
        "```sql",
        "```",  # Code block markers
    ]

    for token in special_tokens:
        sql_text = sql_text.replace(token, "")

    # Extract everything up to and including first semicolon
    if ";" in sql_text:
        sql_text = sql_text.split(";")[0] + ";"

    # Clean up whitespace
    sql_text = " ".join(sql_text.split())

    return sql_text.strip()
