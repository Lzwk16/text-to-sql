"""
Text-to-SQL conversion module using RAG and LLM.

This module provides classes and utilities for converting natural language
queries into SQL queries using Retrieval-Augmented Generation (RAG) and
executing them against SQLite databases with automatic currency normalization.
"""

import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from src.currency_normalizer import CurrencyNormalizer
from src.utils import clean_sql_output, detect_foreign_currencies, is_price_column


class RAGDocumentManager:
    """Manages document creation and retrieval for RAG pipeline.

    This class handles:
    - Converting database tables to document format
    - Creating hybrid retrievers (FAISS + BM25)
    - Managing embeddings and retrieval
    """

    @staticmethod
    def sql_to_documents(db_path: str, sample_limit: int = 5) -> List[Document]:
        """
        Convert SQL Database to documents with context.

        Creates one document per table with:
        - Table name and column names
        - Total record count
        - Sample records

        Args:
            db_path: Path to SQLite database
            sample_limit: Number of sample records to include per table

        Returns:
            List of Document objects
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        documents = []

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]

            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]

            # Get table data
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()

            # Create table overview document
            table_content = f"Table: {table_name}\n"
            table_content += f"Columns: {', '.join(column_names)}\n"
            table_content += f"Total Records: {len(rows)}\n\n"

            # Add sample records
            table_content += "Sample Records:\n"
            for row in rows[:sample_limit]:
                record = dict(zip(column_names, row))
                table_content += f"{record}\n"

            doc = Document(
                page_content=table_content,
                metadata={
                    "source": db_path,
                    "table_name": table_name,
                    "num_records": len(rows),
                    "data_type": "sql_table",
                },
            )
            documents.append(doc)

        conn.close()
        return documents

    @staticmethod
    def create_hybrid_retriever(
        documents: List[Document],
        embedding_model,
        dense_weight: float,
        sparse_weight: float,
    ) -> EnsembleRetriever:
        """
        Create hybrid retriever combining FAISS (dense) and BM25 (sparse).

        Args:
            documents: List of documents to index
            embedding_model: HuggingFace embedding model
            dense_weight: Weight for FAISS semantic search
            sparse_weight: Weight for BM25 keyword search

        Returns:
            EnsembleRetriever combining both retrievers
        """
        # Dense retriever (FAISS)
        dense_vectorstore = FAISS.from_documents(documents, embedding_model)
        dense_retriever = dense_vectorstore.as_retriever()

        # Sparse retriever (BM25)
        sparse_retriever = BM25Retriever.from_documents(documents)

        # Ensemble retriever
        hybrid_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[dense_weight, sparse_weight],
        )

        return hybrid_retriever


class SQLPromptTemplate:
    """Main class for converting natural language to SQL queries using RAG.

    This is the primary interface for the text-to-SQL system. It orchestrates
    the RAG pipeline from natural language input to SQL execution with automatic
    currency normalization.

    Attributes:
        model: Ollama LLM instance
        query: Natural language query string
        use_rag: Whether to use RAG for query generation
        rag_config: Configuration for RAG pipeline
        currency_config: Configuration for currency conversion
        retriever: Hybrid retriever for RAG
        embedding_model: Embedding model for RAG
        documents: Database documents for retrieval
        db_path: Path to database
    """

    def __init__(
        self,
        query: str,
        llm_config: Optional[Dict[str, Any]] = None,
        use_rag: bool = True,
        rag_config: Optional[Dict[str, Any]] = None,
        currency_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize with LLM model and natural language query.

        Args:
            query: Natural language query to process
            llm_config: LLM configuration dict (model_name, model_provider, temperature)
            use_rag: Whether to use RAG for query generation (default: True)
            rag_config: RAG configuration dict
            currency_config: Currency configuration dict

        Raises:
            ValueError: If required config parameters are missing
        """
        self.query = query
        self.use_rag = use_rag

        # Validate and initialize LLM
        llm_validated_config = self._validate_llm_config(llm_config)

        # Initialize LLM with optional temperature
        llm_init_params = {
            "model": llm_validated_config["model_name"],
            "model_provider": llm_validated_config["model_provider"],
        }
        if "temperature" in llm_validated_config:
            llm_init_params["temperature"] = llm_validated_config["temperature"]

        self.model = init_chat_model(**llm_init_params, seed=42)

        # Validate configurations
        self.rag_config = self._validate_rag_config(rag_config) if use_rag else None
        self.currency_config = self._validate_currency_config(currency_config)

        # RAG components (initialized on demand)
        self.retriever: Optional[EnsembleRetriever] = None
        self.embedding_model = None
        self.documents: Optional[List[Document]] = None
        self.db_path: Optional[str] = None

    def _validate_llm_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate LLM configuration parameters.

        Args:
            config: LLM configuration dictionary

        Returns:
            Validated configuration dictionary

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not config:
            raise ValueError(
                "llm_config is required. "
                "Required keys: model_name, model_provider; Optional: temperature"
            )

        # Required parameters
        model_name = config.get("model_name")
        model_provider = config.get("model_provider")

        if not model_name:
            raise ValueError("llm_config must include 'model_name'")
        if not model_provider:
            raise ValueError("llm_config must include 'model_provider'")

        validated_config = {
            "model_name": model_name,
            "model_provider": model_provider,
        }

        # Optional parameter
        temperature = config.get("temperature")
        if temperature is not None:
            validated_config["temperature"] = temperature

        return validated_config

    def _validate_rag_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate RAG configuration parameters.

        Args:
            config: RAG configuration dictionary

        Returns:
            Validated configuration dictionary

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not config:
            raise ValueError(
                "rag_config is required when use_rag=True. "
                "Required keys: rag parameters not found!"
            )

        required_keys = [
            "dense_weight",
            "sparse_weight",
            "k",
            "sample_limit",
            "embedding_model",
        ]
        validated_config = {}

        for key in required_keys:
            value = config.get(key)
            if value is None:
                raise ValueError(f"Missing required RAG config parameter: {key}")
            validated_config[key] = value

        return validated_config

    def _validate_currency_config(
        self, config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate currency configuration parameters.

        Args:
            config: Currency configuration dictionary

        Returns:
            Validated configuration dictionary

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not config:
            raise ValueError(
                "currency_config is required. "
                "Required keys: base_currency, conversion_rates"
            )

        base_currency = config.get("base_currency")
        conversion_rates = config.get("conversion_rates")

        if not base_currency:
            raise ValueError("currency_config must include 'base_currency'")
        if not conversion_rates:
            raise ValueError("currency_config must include 'conversion_rates'")

        return {
            "base_currency": base_currency,
            "conversion_rates": conversion_rates,
        }

    def initialize_rag(self, db_path: str) -> None:
        """Initialize RAG components (documents, embeddings, retriever).

        Args:
            db_path: Path to SQLite database file
        """
        if not self.use_rag:
            return

        print("🔄 Initializing RAG components...")

        self.db_path = db_path

        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.rag_config["embedding_model"]
        )

        # Create documents from database
        self.documents = RAGDocumentManager.sql_to_documents(
            db_path, sample_limit=self.rag_config["sample_limit"]
        )
        print(f"   Created {len(self.documents)} document embeddings")

        # Create hybrid retriever
        self.retriever = RAGDocumentManager.create_hybrid_retriever(
            self.documents,
            self.embedding_model,
            dense_weight=self.rag_config["dense_weight"],
            sparse_weight=self.rag_config["sparse_weight"],
        )
        print("   ✅ RAG initialization complete!")

    def text_to_query(
        self, prompt_template: str, enable_normalization: bool = True
    ) -> Tuple[str, pd.DataFrame]:
        """
        Convert natural language to SQL and execute query with RAG and normalization.

        Args:
            prompt_template: LLM prompt template string
            enable_normalization: Whether to enable automatic currency normalization

        Returns:
            Tuple of (SQL query, results DataFrame)

        Raises:
            ValueError: If RAG not initialized or query generation fails
        """
        if self.use_rag and not self.retriever:
            raise ValueError("RAG not initialized. Call initialize_rag(db_path) first.")

        # Retrieve relevant documents
        k = self.rag_config["k"]
        retrieved_docs = self.retriever.invoke(self.query)[:k]

        # Build schema context from retrieved documents
        db_schema = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generate SQL using LLM
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.model
        response = chain.invoke({"question": self.query, "db_schema": db_schema})

        # Extract content from AIMessage object
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Clean SQL query
        sql_query = clean_sql_output(response_text)

        if not sql_query:
            raise ValueError(
                "Could not extract a valid SQL query from the model's response"
            )

        print("Generated SQL Query:")
        print(sql_query)

        # Execute query
        df = self._execute_sql(sql_query)

        if df is None or df.empty:
            return sql_query, df

        # Check for foreign currencies and normalize if needed
        if enable_normalization:
            df, normalized = self._handle_currency_normalization(df)
            if normalized:
                # Re-execute query after normalization
                print("\n🔄 Re-executing query with normalized data...")
                df = self._execute_sql(sql_query)
                print("✅ Query executed with normalized data!")

        return sql_query, df

    def _execute_sql(self, sql_query: str) -> Optional[pd.DataFrame]:
        """Execute SQL query and return results as DataFrame.

        Args:
            sql_query: SQL query string to execute

        Returns:
            DataFrame containing query results, or None if error
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(sql_query)
            rows = cursor.fetchall()

            if not rows:
                print("⚠️  Query executed successfully, but no data was returned.")
                return None

            columns = [description[0] for description in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
            return df

        except Exception as e:
            print(f"❌ Error executing SQL: {e}")
            return None

        finally:
            cursor.close()
            conn.close()

    def _handle_currency_normalization(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, bool]:
        """Handle currency normalization if foreign currencies detected.

        Args:
            df: Query results DataFrame

        Returns:
            Tuple of (updated DataFrame, whether normalization occurred)
        """
        # Check for price columns
        price_columns = [
            col
            for col in df.columns
            if "price" in col.lower() or is_price_column(df[col])
        ]

        if not price_columns:
            return df, False

        # Check for foreign currencies
        foreign_detected = detect_foreign_currencies(
            df, price_columns, self.currency_config["base_currency"]
        )

        if not foreign_detected:
            return df, False

        print("\n" + "=" * 80)
        print("🌍 FOREIGN CURRENCIES DETECTED - Starting normalization")
        print("=" * 80)

        # Normalize database
        CurrencyNormalizer.normalize_database(
            self.db_path, self.currency_config["conversion_rates"]
        )

        # Re-embed database
        print("\n🔄 Re-embedding database with normalized data...")
        self.documents = RAGDocumentManager.sql_to_documents(
            self.db_path, sample_limit=self.rag_config["sample_limit"]
        )
        print(f"   Created {len(self.documents)} document embeddings")

        self.retriever = RAGDocumentManager.create_hybrid_retriever(
            self.documents,
            self.embedding_model,
            dense_weight=self.rag_config["dense_weight"],
            sparse_weight=self.rag_config["sparse_weight"],
        )
        print("   ✅ Re-embedding complete!")

        return df, True
