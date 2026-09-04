import chromadb
from chromadb.utils import embedding_functions
from app.config import settings

import chromadb
from chromadb.utils import embedding_functions
from app.config import settings

COLLECTION_NAME = "assistant_docs"


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)

        # sentence-transformers embedding function runs locally — no extra API calls/cost
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        """Add or update chunks in the vector store."""
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, query_text: str, top_k: int = 4) -> list[dict]:
        """Return top_k most relevant chunks with metadata and distance score."""
        results = self.collection.query(query_texts=[query_text], n_results=top_k)

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return chunks

    def delete_by_source(self, source_filename: str):
        """Remove all chunks belonging to a given source file — used for re-ingestion."""
        self.collection.delete(where={"source": source_filename})

    def count(self) -> int:
        return self.collection.count()


vector_store = VectorStore()