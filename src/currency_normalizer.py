"""
Currency normalization module for database tables.

This module provides functionality to:
- Scan databases for price columns
- Normalize multi-currency prices to a base currency
- Handle batch currency conversion across database tables
"""

import sqlite3
from typing import Dict, List

import pandas as pd

from src.utils import convert_price_to_sgd


class CurrencyNormalizer:
    """Handles currency normalization in database tables.

    This class provides methods to:
    - Scan database for price columns
    - Normalize prices to base currency
    - Handle multi-currency data
    """

    @staticmethod
    def scan_for_price_columns(db_path: str) -> Dict[str, List[str]]:
        """
        Scan database for tables containing price columns.

        Args:
            db_path: Path to SQLite database

        Returns:
            Dictionary mapping table_name -> list of price column names
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        tables_with_price_columns = {}

        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()

            for col_info in columns_info:
                col_name = col_info[1]
                if "price" in col_name.lower():
                    if table_name not in tables_with_price_columns:
                        tables_with_price_columns[table_name] = []
                    tables_with_price_columns[table_name].append(col_name)

        conn.close()
        return tables_with_price_columns

    @staticmethod
    def normalize_table_to_base_currency(
        table_name: str,
        price_columns: List[str],
        db_path: str,
        conversion_rates: Dict[str, float],
    ) -> None:
        """
        Normalize all price columns in a table to base currency.

        Args:
            table_name: Name of table to normalize
            price_columns: List of price column names
            db_path: Path to SQLite database
            conversion_rates: Dictionary mapping currency codes to conversion rates
        """
        conn = sqlite3.connect(db_path)

        print(f"\n📊 Normalizing table: '{table_name}'")

        # Load entire table
        table_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        print(f"   Loaded {len(table_df)} rows")

        # Convert each price column
        for col_name in price_columns:
            print(f"   Converting '{col_name}' to base currency...")

            table_df[col_name] = table_df[col_name].apply(
                lambda x: convert_price_to_sgd(x, conversion_rates)
            )

            print(f"   ✅ Converted '{col_name}' to numeric values")

        # Write back to database
        print("   💾 Writing updated data to database...")
        table_df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"   ✅ Updated '{table_name}' in database")

        conn.commit()
        conn.close()

    @staticmethod
    def normalize_database(db_path: str, conversion_rates: Dict[str, float]) -> None:
        """
        Normalize all price columns in database to base currency.

        Args:
            db_path: Path to SQLite database
            conversion_rates: Dictionary mapping currency codes to conversion rates
        """
        print("\n" + "=" * 80)
        print("🔄 NORMALIZING DATABASE")
        print("=" * 80)

        tables_with_price_columns = CurrencyNormalizer.scan_for_price_columns(db_path)

        if not tables_with_price_columns:
            print("No price columns found in database.")
            return

        for table_name, price_cols in tables_with_price_columns.items():
            CurrencyNormalizer.normalize_table_to_base_currency(
                table_name, price_cols, db_path, conversion_rates
            )

        print("\n" + "=" * 80)
        print("✅ DATABASE NORMALIZATION COMPLETE!")
        print("=" * 80)
