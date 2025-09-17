"""
Text-to-SQL conversion module using LLM and database schema analysis.

This module provides classes and utilities for converting natural language
queries into SQL queries and executing them against SQLite databases.
"""

import re
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain.llms import Ollama
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


@dataclass
class ColumnInfo:
    """Information about a database column.
    
    Attributes:
        name: Column name
        type: SQL data type
        nullable: Whether column accepts NULL values
        primary_key: Whether column is a primary key
        samples: Sample values for format detection (TEXT columns only)
    """
    name: str
    type: str
    nullable: bool
    primary_key: bool
    samples: Optional[List[str]] = None


@dataclass
class TableInfo:
    """Information about a database table.
    
    Attributes:
        name: Table name
        columns: List of column information
        primary_key: Name of primary key column (if any)
    """
    name: str
    columns: List[ColumnInfo]
    primary_key: Optional[str] = None


@dataclass
class SchemaAnalysis:
    """Analysis results of database schema.
    
    Attributes:
        tables: Dictionary mapping table names to TableInfo objects
        numeric_columns: List of numeric column identifiers
        categorical_columns: List of categorical column identifiers
        date_columns: List of date/time column identifiers
    """
    tables: Dict[str, TableInfo]
    numeric_columns: List[str]
    categorical_columns: List[str]
    date_columns: List[str]


class DatabaseManager:
    """Handles database connections and schema operations.

    This class manages SQLite database connections and provides methods for
    extracting schema information and executing queries. It maintains a
    connection pool and handles database operations safely.

    Attributes:
        db_path: Path to the SQLite database file
        engine: SQLAlchemy engine instance (created lazily)
    """

    def __init__(self, db_path: str):
        """Initialize database manager with path to SQLite database.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.engine: Optional[Engine] = None

    def connect(self) -> Engine:
        """Create and return database engine.

        Returns:
            SQLAlchemy Engine instance for database operations
        """
        if not self.engine:
            self.engine = create_engine(f"sqlite:///{self.db_path}")
        return self.engine

    def get_table_names(self) -> List[str]:
        """Get list of table names in the database.

        Returns:
            List of table names found in the database
        """
        engine = self.connect()
        inspector = inspect(engine)
        return inspector.get_table_names()

    def get_table_info(self, table_name: str) -> TableInfo:
        """Get detailed information about a specific table.

        Args:
            table_name: Name of the table to analyze

        Returns:
            TableInfo object containing table structure and column details
        """
        engine = self.connect()

        with engine.connect() as conn:
            # Get column information using PRAGMA
            table_info_query = text(f"PRAGMA table_info({table_name})")
            columns_info = conn.execute(table_info_query).fetchall()

            columns = []
            primary_key = None

            for col_data in columns_info:
                col_name = col_data[1]
                col_type = col_data[2]
                is_nullable = not col_data[3]  # notnull is 1 if NOT nullable
                is_pk = col_data[5] == 1

                if is_pk:
                    primary_key = col_name

                # Get sample data for TEXT columns
                samples = None
                if col_type == 'TEXT':
                    samples = self._get_column_samples(conn, table_name, col_name)

                column_info = ColumnInfo(
                    name=col_name,
                    type=col_type,
                    nullable=is_nullable,
                    primary_key=is_pk,
                    samples=samples
                )
                columns.append(column_info)

            return TableInfo(
                name=table_name,
                columns=columns,
                primary_key=primary_key
            )

    def _get_column_samples(
        self, conn, table_name: str, col_name: str, limit: int = 3
    ) -> List[str]:
        """Get sample values from a column for format detection.

        Args:
            conn: Database connection
            table_name: Name of the table
            col_name: Name of the column
            limit: Maximum number of samples to retrieve

        Returns:
            List of sample values as strings
        """
        try:
            sample_query = text(
                f"SELECT DISTINCT {col_name} FROM {table_name} "
                f"WHERE {col_name} IS NOT NULL AND {col_name} <> '' LIMIT {limit}"
            )
            sample_results = conn.execute(sample_query).fetchall()
            return [str(row[0]) for row in sample_results] if sample_results else []
        except Exception:  # pylint: disable=broad-except
            return []

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame.

        Args:
            sql_query: SQL query string to execute

        Returns:
            DataFrame containing query results
        """
        engine = self.connect()
        with engine.connect() as conn:
            return pd.read_sql_query(text(sql_query), conn)


class SchemaExtractor:
    """Extracts and analyzes database schema.

    This class handles the extraction of database schema information,
    including table structures, column types, constraints, and sample
    data for format detection. It provides methods to analyze and
    format schema information for use with LLM prompts.

    Attributes:
        db_manager: DatabaseManager instance for database operations
    """

    def __init__(self, db_manager: DatabaseManager):
        """Initialize schema extractor with database manager.
        
        Args:
            db_manager: DatabaseManager instance to use for operations
        """
        self.db_manager = db_manager

    def extract_full_schema(self) -> SchemaAnalysis:
        """Extract complete schema information from database.

        Returns:
            SchemaAnalysis object containing comprehensive schema information
        """
        table_names = self.db_manager.get_table_names()
        tables = {}
        numeric_columns = []
        categorical_columns = []
        date_columns = []

        for table_name in table_names:
            table_info = self.db_manager.get_table_info(table_name)
            tables[table_name] = table_info

            # Categorize columns
            for col in table_info.columns:
                col_identifier = f"{table_name}.{col.name}"

                if self._is_numeric_type(col.type):
                    numeric_columns.append(col_identifier)
                elif self._is_date_type(col.type):
                    date_columns.append(col_identifier)
                else:
                    categorical_columns.append(col_identifier)

        return SchemaAnalysis(
            tables=tables,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            date_columns=date_columns
        )

    def format_schema_for_prompt(self, schema: SchemaAnalysis) -> str:
        """Format schema information for LLM prompt.

        Args:
            schema: SchemaAnalysis object to format

        Returns:
            Formatted string representation of schema for LLM consumption
        """
        formatted_parts = []

        for table_name, table_info in schema.tables.items():
            table_parts = [f"Table: {table_name}"]

            for col in table_info.columns:
                # Build constraint information
                constraints = []
                if col.primary_key:
                    constraints.append("PRIMARY KEY")
                if not col.nullable:
                    constraints.append("NOT NULL")

                constraint_str = f" ({', '.join(constraints)})" if constraints else ""

                # Add sample data for TEXT columns
                sample_str = ""
                if col.samples:
                    sample_str = f" [samples: {', '.join(col.samples)}]"

                table_parts.append(
                    f"  - {col.name} ({col.type}){constraint_str}{sample_str}"
                )

            formatted_parts.extend(table_parts)
            formatted_parts.append("")  # Blank line between tables

        return "\n".join(formatted_parts)

    @staticmethod
    def _is_numeric_type(col_type: str) -> bool:
        """Check if column type is numeric.

        Args:
            col_type: Database column type string

        Returns:
            True if column type represents numeric data
        """
        col_type_lower = col_type.lower()
        return any(
            numeric_type in col_type_lower
            for numeric_type in ["int", "float", "double", "decimal", "real"]
        )

    @staticmethod
    def _is_date_type(col_type: str) -> bool:
        """Check if column type is date/time related.

        Args:
            col_type: Database column type string

        Returns:
            True if column type represents date/time data
        """
        col_type_lower = col_type.lower()
        return any(
            date_type in col_type_lower
            for date_type in ["date", "time", "timestamp"]
        )


class QueryAnalyzer:
    """Analyzes SQL queries to understand their structure and purpose.

    This utility class provides static methods for parsing and analyzing
    SQL queries to understand their components, structure, and intent.
    Used for result formatting and interpretation generation.
    """

    @staticmethod
    def analyze_query_type(sql_query: str) -> Dict[str, Any]:
        """Analyze SQL query structure and components.

        Args:
            sql_query: SQL query string to analyze

        Returns:
            Dictionary containing query analysis results
        """
        sql_lower = sql_query.lower()

        return {
            "is_aggregate": bool(re.search(r"\b(count|sum|avg|max|min)\s*\(", sql_lower)),
            "has_group_by": "group by" in sql_lower,
            "has_order_by": "order by" in sql_lower,
            "has_limit": bool(re.search(r"\blimit\s+\d+", sql_lower)),
            "aggregate_functions": re.findall(
                r"\b(count|sum|avg|max|min)\s*\(([^)]+)\)", sql_lower
            ),
            "selected_columns": QueryAnalyzer._extract_selected_columns(sql_lower),
        }

    @staticmethod
    def _extract_selected_columns(sql_query: str) -> List[str]:
        """Extract column names from SELECT clause.

        Args:
            sql_query: SQL query string

        Returns:
            List of selected column names
        """
        match = re.search(r"select\s+(.*?)\s+from", sql_query, re.IGNORECASE)
        if not match:
            return []

        select_clause = match.group(1)
        columns = []

        for col in select_clause.split(","):
            col = col.strip()
            # Handle aliased columns
            if " as " in col.lower():
                col = col.split(" as ")[1]
            # Remove function wrappers
            col = re.sub(r"\b(count|sum|avg|max|min)\s*\(([^)]+)\)", r"\2", col)
            columns.append(col.strip("\"'"))

        return columns


class ResultFormatter:
    """Formats query results and generates interpretations.

    This class handles formatting of query results for display and
    generates natural language interpretations of the results based
    on query type and content. Includes monetary formatting and
    intelligent result summarization.
    """

    @staticmethod
    def format_results(results: pd.DataFrame, sql_query: str) -> Tuple[pd.DataFrame, str]:
        """Format results DataFrame and generate interpretation.

        Args:
            results: Raw query results as DataFrame
            sql_query: Original SQL query string

        Returns:
            Tuple of formatted DataFrame and natural language interpretation
        """
        if results.empty:
            return results, "No results found for your query."

        # Format numeric columns
        formatted_results = results.copy()
        for col in results.select_dtypes(include=["float64", "int64"]).columns:
            if ResultFormatter._is_monetary_column(col):
                formatted_results[col] = results[col].apply(lambda x: f"${x:,.2f}")
            else:
                formatted_results[col] = results[col].apply(lambda x: f"{x:,}")

        # Generate interpretation
        query_analysis = QueryAnalyzer.analyze_query_type(sql_query)
        interpretation = ResultFormatter._generate_interpretation(
            results, query_analysis
        )

        return formatted_results, interpretation

    @staticmethod
    def _is_monetary_column(col_name: str) -> bool:
        """Check if column represents monetary values.

        Args:
            col_name: Name of the column to check

        Returns:
            True if column appears to contain monetary values
        """
        return any(
            suffix in col_name.lower()
            for suffix in ["_price", "_amount", "_cost", "price"]
        )

    @staticmethod
    def _generate_interpretation(
        results: pd.DataFrame,
        query_type: Dict[str, Any]
    ) -> str:
        """Generate natural language interpretation of results.

        Args:
            results: Query results DataFrame
            query_type: Query analysis results

        Returns:
            Natural language interpretation string
        """
        if len(results) == 1:
            return ResultFormatter._interpret_single_result(results, query_type)
        if query_type["has_group_by"]:
            return ResultFormatter._interpret_grouped_results(results, query_type)
        return ResultFormatter._interpret_multiple_results(results, query_type)

    @staticmethod
    def _interpret_single_result(results: pd.DataFrame, query_type: Dict[str, Any]) -> str:
        """Interpret single-row results.

        Args:
            results: Single-row DataFrame
            query_type: Query analysis results

        Returns:
            Interpretation string for single result
        """
        if query_type["is_aggregate"]:
            interpretations = []
            for col in results.columns:
                value = results.iloc[0][col]
                if ResultFormatter._is_monetary_column(col):
                    interpretations.append(
                        f"The {col.replace('_', ' ')} is {value}"
                    )
                else:
                    interpretations.append(f"{col.replace('_', ' ')}: {value}")
            return " | ".join(interpretations)

        # Non-aggregate single result
        parts = []
        for col in results.columns:
            value = results.iloc[0][col]
            if ResultFormatter._is_monetary_column(col):
                parts.append(f"with {col.replace('_', ' ')} of {value}")
            else:
                parts.append(str(value))
        return f"Found: {' '.join(parts)}"

    @staticmethod
    def _interpret_grouped_results(results: pd.DataFrame, query_type: Dict[str, Any]) -> str:
        """Interpret grouped results.

        Args:
            results: Multi-row grouped DataFrame
            query_type: Query analysis results

        Returns:
            Interpretation string for grouped results
        """
        group_col = query_type["selected_columns"][0] if query_type["selected_columns"] else "items"
        return f"Showing results grouped by {group_col} ({len(results)} groups)"

    @staticmethod
    def _interpret_multiple_results(results: pd.DataFrame, query_type: Dict[str, Any]) -> str:
        """Interpret multiple-row results.

        Args:
            results: Multi-row DataFrame
            query_type: Query analysis results

        Returns:
            Interpretation string for multiple results
        """
        if query_type["has_order_by"]:
            monetary_cols = [
                col for col in results.columns
                if ResultFormatter._is_monetary_column(col)
            ]
            if monetary_cols:
                return f"Showing {len(results)} results ordered by {monetary_cols[0]}"
        return f"Found {len(results)} matching records"


class SQLQueryCleaner:
    """Cleans and validates SQL queries from LLM responses.

    This utility class handles the extraction and cleaning of SQL queries
    from potentially messy LLM responses that may contain explanations,
    markdown formatting, or other non-SQL content.
    """

    @staticmethod
    def extract_sql_query(text_response: str) -> str:
        """Extract clean SQL query from LLM response.

        Args:
            text_response: Raw response text from LLM

        Returns:
            Cleaned SQL query string
        """
        # Remove content within tags
        text_response = re.sub(r"<[^>]+>.*?</[^>]+>", "", text_response, flags=re.DOTALL)

        # Process line by line
        sql_parts = []
        for line in text_response.split("\n"):
            line = line.strip()
            if SQLQueryCleaner._should_skip_line(line):
                continue
            sql_parts.append(line)

        # Join and clean
        sql_query = " ".join(sql_parts).strip()
        sql_query = re.sub(r"```sql|```", "", sql_query)
        sql_query = re.sub(r"this query.*$", "", sql_query, flags=re.DOTALL)

        return sql_query.strip()

    @staticmethod
    def _should_skip_line(line: str) -> bool:
        """Check if a line should be skipped during SQL extraction.

        Args:
            line: Line of text to evaluate

        Returns:
            True if line should be skipped
        """
        if not line:
            return True

        skip_prefixes = [
            "this query", "--", "#", "```", "let", "i ", "the ", "here", "now"
        ]
        return any(line.lower().startswith(prefix) for prefix in skip_prefixes)


class SQLPromptTemplate:
    """Main class for converting natural language to SQL queries.

    This is the primary interface for the text-to-SQL system. It orchestrates
    the entire process from natural language input to SQL execution and result
    formatting. The class maintains backward compatibility while leveraging
    the modular component architecture.

    Attributes:
        model: Ollama LLM instance
        query: Natural language query string
        db_manager: Database connection manager
        schema_extractor: Schema analysis component
        schema_info: Cached schema analysis results
    """

    def __init__(self, model_name: str, query: str):
        """Initialize with LLM model and natural language query.

        Args:
            model_name: Name of the Ollama model to use
            query: Natural language query to process
        """
        self.model = Ollama(model=model_name)
        self.query = query
        self.db_manager: Optional[DatabaseManager] = None
        self.schema_extractor: Optional[SchemaExtractor] = None
        self.schema_info: Optional[SchemaAnalysis] = None

    def extract_schema(self, db_path: str) -> str:
        """Extract database schema and return formatted string for prompt.

        Args:
            db_path: Path to SQLite database file

        Returns:
            Formatted schema string for LLM prompt
        """
        self.db_manager = DatabaseManager(db_path)
        self.schema_extractor = SchemaExtractor(self.db_manager)

        # Extract and store schema analysis
        self.schema_info = self.schema_extractor.extract_full_schema()

        # Return formatted schema for prompt
        return self.schema_extractor.format_schema_for_prompt(self.schema_info)

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query using database manager.

        Args:
            sql_query: SQL query string to execute

        Returns:
            DataFrame containing query results

        Raises:
            ValueError: If database connection not initialized
        """
        if not self.db_manager:
            raise ValueError("Database connection not initialized. Call extract_schema first.")
        return self.db_manager.execute_query(sql_query)

    def text_to_query(self, schema: str, prompt_template: str) -> Tuple[str, pd.DataFrame, str]:
        """Convert natural language to SQL and execute query.

        Args:
            schema: Formatted schema string
            prompt_template: LLM prompt template

        Returns:
            Tuple of (SQL query, formatted results, interpretation)

        Raises:
            ValueError: If SQL extraction fails
        """
        # Generate SQL using LLM
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.model
        response_text = chain.invoke({"query": self.query, "schema": schema})

        # Extract and clean SQL query
        sql_query = SQLQueryCleaner.extract_sql_query(response_text)
        if not sql_query:
            raise ValueError("Could not extract a valid SQL query from the model's response")

        # Execute query and format results
        raw_results = self.execute_query(sql_query)
        formatted_results, interpretation = ResultFormatter.format_results(raw_results, sql_query)

        return sql_query, formatted_results, interpretation

    # Legacy methods for backward compatibility
    def analyze_schema(self, schema_dict: Dict) -> Dict[str, Any]:  # pylint: disable=unused-argument
        """Legacy method - convert SchemaAnalysis to old format.

        Args:
            schema_dict: Unused parameter for backward compatibility

        Returns:
            Schema analysis in legacy dictionary format

        Raises:
            ValueError: If schema not extracted
        """
        if not self.schema_info:
            raise ValueError("Schema not extracted. Call extract_schema first.")

        return {
            "tables": {
                name: {
                    "columns": {
                        col.name: {"type": col.type, "is_primary_key": col.primary_key}
                        for col in info.columns
                    },
                    "primary_key": info.primary_key
                }
                for name, info in self.schema_info.tables.items()
            },
            "numeric_columns": self.schema_info.numeric_columns,
            "categorical_columns": self.schema_info.categorical_columns,
            "date_columns": self.schema_info.date_columns,
        }

    def format_results(self, results: pd.DataFrame, sql_query: str) -> Tuple[pd.DataFrame, str]:
        """Legacy method - delegate to ResultFormatter.

        Args:
            results: Query results DataFrame
            sql_query: Original SQL query

        Returns:
            Tuple of formatted DataFrame and interpretation
        """
        return ResultFormatter.format_results(results, sql_query)