"""Per-database schema context for the LLM.

Small databases get the full schema. Larger ones get the top-k tables from a
hybrid FAISS + BM25 retriever, expanded with foreign-key neighbours so join
tables are never dropped. Every index is private to one database.
"""

import threading
from typing import Dict, List, Optional

from utils.schema import TableInfo, introspect


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
        sparse = BM25Retriever.from_documents(docs)
        sparse.k = self.k
        return EnsembleRetriever(retrievers=[dense, sparse], weights=self.weights)

    def schema_for(self, db_id: str, db_path: str, question: str) -> str:
        """Schema text relevant to ``question`` for the given database."""
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
        return "\n\n".join(t.render() for t in selected)

    def forget(self, db_id: Optional[str]) -> None:
        with self._lock:
            self._tables.pop(db_id, None)
            self._retrievers.pop(db_id, None)
