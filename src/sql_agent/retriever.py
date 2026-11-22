"""Per-database schema context for the LLM.

Small databases get the full schema. Larger ones get the top-k tables from a
hybrid FAISS + BM25 retriever, expanded with foreign-key neighbours so join
tables are never dropped. Every index is private to one database.
"""

import re
import threading
from typing import Dict, List, Optional

from src.sql_agent.postprocess.base import Converter, Detected
from src.sql_agent.postprocess.detect import detect_columns
from utils.schema import TableInfo, introspect


_WORD = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens; splits quotes, underscores and camelCase.

    BM25's default tokenizer splits on whitespace only, so a quoted DDL name like
    ``"Cabin Comfort"`` becomes the tokens ``"Cabin`` and ``Comfort"``, which never
    match the words in a question.
    """
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [w.lower() for w in _WORD.findall(spaced)]


class SchemaProvider:
    def __init__(
        self,
        sample_limit: int = 3,
        full_schema_max_tables: int = 8,
        k: int = 5,
        dense_weight: float = 0.3,
        sparse_weight: float = 0.7,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.sample_limit = sample_limit
        self.full_schema_max_tables = full_schema_max_tables
        self.k = k
        self.weights = [dense_weight, sparse_weight]
        self.embedding_model = embedding_model
        self._tables: Dict[str, List[TableInfo]] = {}
        self._retrievers: Dict[str, object] = {}
        self._detected: Dict[tuple, List[Detected]] = {}
        self._embeddings = None
        self._lock = threading.Lock()

    def tables(self, db_id: str, db_path: str) -> List[TableInfo]:
        with self._lock:
            if db_id not in self._tables:
                self._tables[db_id] = introspect(db_path, self.sample_limit)
            return self._tables[db_id]

    def _build_retriever(self, db_id: str, tables: List[TableInfo]):
        # Imported lazily: heavy, and only needed for large schemas.
        from langchain_classic.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        from langchain_huggingface import HuggingFaceEmbeddings

        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        docs = [
            Document(page_content=t.render(), metadata={"table": t.name})
            for t in tables
        ]
        dense = FAISS.from_documents(docs, self._embeddings).as_retriever(
            search_kwargs={"k": self.k}
        )
        sparse = BM25Retriever.from_documents(docs, preprocess_func=tokenize)
        sparse.k = self.k
        return EnsembleRetriever(retrievers=[dense, sparse], weights=self.weights)

    def convertible(
        self, db_id: str, db_path: str, converters: List[Converter]
    ) -> List[Detected]:
        """Columns that some converter recognises; scanned once per database."""
        key = (db_id, tuple(c.cache_key for c in converters))
        with self._lock:
            if key not in self._detected:
                self._detected[key] = detect_columns(db_path, converters)
            return self._detected[key]

    def schema_for(self, db_id: str, db_path: str, question: str) -> str:
        """Schema text relevant to ``question`` for the given database."""
        return "\n\n".join(
            t.render() for t in self.select_tables(db_id, db_path, question)
        )

    def select_tables(
        self, db_id: str, db_path: str, question: str
    ) -> List[TableInfo]:
        """The tables shown to the model for ``question``."""
        tables = self.tables(db_id, db_path)
        if len(tables) <= self.full_schema_max_tables:
            selected = tables
        else:
            with self._lock:
                retriever = self._retrievers.get(db_id)
                if retriever is None:
                    retriever = self._build_retriever(db_id, tables)
                    self._retrievers[db_id] = retriever
            hits = retriever.invoke(question)[: self.k]
            by_name = {t.name: t for t in tables}
            names = [d.metadata["table"] for d in hits]
            for n in list(names):  # pull in FK neighbours (one hop, both ways)
                names.extend(by_name[n].referenced_tables)
                names.extend(
                    t.name for t in tables if n in t.referenced_tables
                )
            selected = [by_name[n] for n in dict.fromkeys(names) if n in by_name]
        return selected

    def is_partial(self, db_id: str, db_path: str) -> bool:
        """True when ``schema_for`` sends only a subset of the tables."""
        return len(self.tables(db_id, db_path)) > self.full_schema_max_tables

    def full_schema(self, db_id: str, db_path: str, max_chars: int = 30000) -> Optional[str]:
        """Every table, or None if that would be too large for a prompt."""
        text = "\n\n".join(t.render() for t in self.tables(db_id, db_path))
        return text if len(text) <= max_chars else None

    def forget(self, db_id: Optional[str]) -> None:
        with self._lock:
            self._tables.pop(db_id, None)
            self._retrievers.pop(db_id, None)
            for key in [k for k in self._detected if k[0] == db_id]:
                del self._detected[key]
