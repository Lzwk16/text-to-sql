import json
import re
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from sqlalchemy import create_engine, inspect, text


class SQLPromptTemplate:
    def __init__(self, model_name: str, query: str):
        """
        Initialize the SQL prompt template.

        Parameters
        ----------
        model_name : str
            The name of the language model to use.
        query : str
            The natural language query to process.
        """
        self.model = OllamaLLM(model=model_name)
        self.query = query
        self.engine = None
        self.schema_info = None

    def analyze_schema(self, schema_dict: Dict) -> Dict[str, Any]:
        """
        Analyze the database schema to understand its structure and relationships.

        Parameters
        ----------
        schema_dict : Dict
            Dictionary containing table and column information.

        Returns
        -------
        Dict[str, Any]
            Analysis of the schema including data types, relationships, etc.
        """
        analysis = {
            "tables": {},
            "relationships": [],
            "numeric_columns": [],
            "categorical_columns": [],
            "date_columns": [],
        }

        with self.engine.connect() as conn:
            for table, columns in schema_dict.items():
                # Use text() for safe SQL execution
                table_info_query = text(f"PRAGMA table_info({table})")
                table_info = conn.execute(table_info_query).fetchall()

                analysis["tables"][table] = {"columns": {}, "primary_key": None}

                for col_info in table_info:
                    col_name = col_info[1]
                    col_type = col_info[2].lower()
                    is_pk = col_info[5] == 1

                    if is_pk:
                        analysis["tables"][table]["primary_key"] = col_name

                    # Categorize columns
                    if (
                        "int" in col_type
                        or "float" in col_type
                        or "double" in col_type
                        or "decimal" in col_type
                    ):
                        analysis["numeric_columns"].append(f"{table}.{col_name}")
                    elif "date" in col_type or "time" in col_type:
                        analysis["date_columns"].append(f"{table}.{col_name}")
                    else:
                        analysis["categorical_columns"].append(f"{table}.{col_name}")

                    analysis["tables"][table]["columns"][col_name] = {
                        "type": col_type,
                        "is_primary_key": is_pk,
                    }

        return analysis

    def extract_schema(self, db_path: str) -> str:
        """
        Extract and analyze the schema from the database.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file.

        Returns
        -------
        str
            JSON string of the schema.
        """
        self.engine = create_engine(f"sqlite:///{db_path}")
        inspector = inspect(self.engine)
        schema = {}

        # Get detailed schema information including column types
        with self.engine.connect() as conn:
            for table_name in inspector.get_table_names():
                # Get column information using PRAGMA
                table_info_query = text(f"PRAGMA table_info({table_name})")
                columns_info = conn.execute(table_info_query).fetchall()

                # Format column information
                columns = []
                for col in columns_info:
                    col_name = col[1]
                    col_type = col[2]
                    is_nullable = not col[
                        3
                    ]  # notnull is 1 if the column is NOT nullable
                    is_pk = col[5] == 1

                    col_info = {
                        "name": col_name,
                        "type": col_type,
                        "nullable": is_nullable,
                        "primary_key": is_pk,
                    }
                    columns.append(col_info)

                schema[table_name] = columns

        # Store schema analysis for result interpretation
        self.schema_info = self.analyze_schema(schema)

        # Format schema for the prompt
        formatted_schema = []
        for table_name, columns in schema.items():
            table_info = [f"Table: {table_name}"]
            for col in columns:
                constraints = []
                if col["primary_key"]:
                    constraints.append("PRIMARY KEY")
                if not col["nullable"]:
                    constraints.append("NOT NULL")

                constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                table_info.append(f"  - {col['name']} ({col['type']}){constraint_str}")

            formatted_schema.extend(table_info)
            formatted_schema.append("")  # Add blank line between tables

        return "\n".join(formatted_schema)

    def format_results(
        self, results: pd.DataFrame, sql_query: str
    ) -> Tuple[pd.DataFrame, str]:
        """
        Format the query results based on the query type and schema information.

        Parameters
        ----------
        results : pd.DataFrame
            The query results as a DataFrame.
        sql_query : str
            The SQL query that generated the results.

        Returns
        -------
        Tuple[pd.DataFrame, str]
            Formatted DataFrame and a natural language interpretation of the results.
        """
        if results.empty:
            return results, "No results found for your query."

        # Analyze the query type
        query_type = self._analyze_query_type(sql_query.lower())

        # Format numeric columns
        for col in results.select_dtypes(include=["float64", "int64"]).columns:
            if any(
                col.lower().endswith(suffix)
                for suffix in ["_price", "_amount", "_cost", "price"]
            ):
                results[col] = results[col].apply(lambda x: f"${x:,.2f}")
            else:
                results[col] = results[col].apply(lambda x: f"{x:,}")

        # Generate interpretation based on query type and results
        interpretation = self._generate_interpretation(results, query_type, sql_query)

        return results, interpretation

    def _analyze_query_type(self, sql_query: str) -> Dict[str, Any]:
        """
        Analyze the type of SQL query to determine how to format results.

        Parameters
        ----------
        sql_query : str
            The SQL query to analyze.

        Returns
        -------
        Dict[str, Any]
            Analysis of the query type and its components.
        """
        analysis = {
            "is_aggregate": bool(
                re.search(r"\b(count|sum|avg|max|min)\s*\(", sql_query)
            ),
            "has_group_by": "group by" in sql_query,
            "has_order_by": "order by" in sql_query,
            "has_limit": bool(re.search(r"\blimit\s+\d+", sql_query)),
            "aggregate_functions": re.findall(
                r"\b(count|sum|avg|max|min)\s*\(([^)]+)\)", sql_query
            ),
            "selected_columns": self._extract_selected_columns(sql_query),
        }
        return analysis

    def _extract_selected_columns(self, sql_query: str) -> List[str]:
        """
        Extract the columns being selected in the query.

        Parameters
        ----------
        sql_query : str
            The SQL query to analyze.

        Returns
        -------
        List[str]
            List of selected column names.
        """
        # Extract everything between SELECT and FROM
        match = re.search(r"select\s+(.*?)\s+from", sql_query, re.IGNORECASE)
        if not match:
            return []

        select_clause = match.group(1)
        # Split by comma and clean up each column
        columns = []
        for col in select_clause.split(","):
            col = col.strip()
            # Handle aliased columns
            if " as " in col.lower():
                col = col.split(" as ")[1]
            # Remove any functions wrapping the column
            col = re.sub(r"\b(count|sum|avg|max|min)\s*\(([^)]+)\)", r"\2", col)
            columns.append(col.strip("\"'"))
        return columns

    def _generate_interpretation(
        self, results: pd.DataFrame, query_type: Dict[str, Any], sql_query: str
    ) -> str:
        """
        Generate a natural language interpretation of the results.

        Parameters
        ----------
        results : pd.DataFrame
            The query results.
        query_type : Dict[str, Any]
            Analysis of the query type.
        sql_query : str
            The original SQL query.

        Returns
        -------
        str
            Natural language interpretation of the results.
        """
        if len(results) == 1:
            if query_type["is_aggregate"]:
                # Handle aggregate queries
                interpretations = []
                for col in results.columns:
                    value = results.iloc[0][col]
                    # Check if this is a price/monetary value
                    if any(
                        price_term in col.lower()
                        for price_term in ["price", "amount", "cost"]
                    ):
                        interpretations.append(
                            f"The {col.replace('_', ' ')} is {value}"
                        )
                    else:
                        interpretations.append(f"{col.replace('_', ' ')}: {value}")
                return " | ".join(interpretations)

            elif query_type["has_order_by"] and query_type["has_limit"]:
                # Handle "most/least" type queries
                parts = []
                for col in results.columns:
                    value = results.iloc[0][col]
                    if any(
                        price_term in col.lower()
                        for price_term in ["price", "amount", "cost"]
                    ):
                        parts.append(f"with {col.replace('_', ' ')} of {value}")
                    else:
                        parts.append(str(value))
                return f"Found: {' '.join(parts)}"

        elif query_type["has_group_by"]:
            # Handle grouped results
            group_col = (
                query_type["selected_columns"][0]
                if query_type["selected_columns"]
                else "items"
            )
            return f"Showing results grouped by {group_col} ({len(results)} groups)"

        elif len(results) > 1:
            if query_type["has_order_by"]:
                order_col = next(
                    (
                        col
                        for col in results.columns
                        if any(
                            term in col.lower() for term in ["price", "amount", "cost"]
                        )
                    ),
                    None,
                )
                if order_col:
                    return f"Showing {len(results)} results ordered by {order_col}"
            return f"Found {len(results)} matching records"

        return "Query executed successfully"

    def extract_sql_query(self, text_response: str) -> str:
        """
        Extract the SQL query from the model's response.

        Parameters
        ----------
        text_response : str
            The raw response from the language model.

        Returns
        -------
        str
            The cleaned SQL query.
        """
        # Remove any content within tags
        text_response = re.sub(
            r"<[^>]+>.*?</[^>]+>", "", text_response, flags=re.DOTALL
        )

        # Split into lines and process
        sql_parts = text_response.split("\n")
        cleaned_parts = []

        for part in sql_parts:
            part = part.strip()
            # Skip non-SQL content
            if not part or any(
                part.lower().startswith(prefix)
                for prefix in [
                    "this query",
                    "--",
                    "#",
                    "```",
                    "let",
                    "i ",
                    "the ",
                    "here",
                    "now",
                ]
            ):
                continue
            cleaned_parts.append(part)

        # Join the parts and clean up
        sql_query = " ".join(cleaned_parts).strip()
        sql_query = re.sub(r"```sql|```", "", sql_query)
        sql_query = re.sub(r"this query.*$", "", sql_query, flags=re.DOTALL)

        return sql_query.strip()

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """
        Execute the SQL query and return formatted results.

        Parameters
        ----------
        sql_query : str
            The SQL query to execute.

        Returns
        -------
        pd.DataFrame
            The query results as a formatted DataFrame.
        """
        if not self.engine:
            raise ValueError(
                "Database connection not initialized. Call extract_schema first."
            )

        with self.engine.connect() as conn:
            return pd.read_sql_query(text(sql_query), conn)

    def text_to_query(
        self, schema: str, prompt_template: str
    ) -> Tuple[str, pd.DataFrame, str]:
        """
        Convert natural language to SQL query and execute it.

        Parameters
        ----------
        schema : str
            The extracted schema of the database.
        prompt_template : str
            The prompt template to use.

        Returns
        -------
        Tuple[str, pd.DataFrame, str]
            The SQL query, formatted results, and result interpretation.
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.model
        response_text = chain.invoke({"query": self.query, "schema": schema})

        # Extract and validate SQL query
        sql_query = self.extract_sql_query(response_text)
        if not sql_query:
            raise ValueError(
                "Could not extract a valid SQL query from the model's response"
            )

        # Execute query and format results
        raw_results = self.execute_query(sql_query)
        formatted_results, interpretation = self.format_results(raw_results, sql_query)

        return sql_query, formatted_results, interpretation
