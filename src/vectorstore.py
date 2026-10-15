import warnings
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from src.config import CHROMA_PERSIST_DIR, DEFAULT_EMBEDDING_MODEL


class FastLocalEmbeddings(Embeddings):
    """Lightweight, resilient local text embeddings using normalized hashing vectorization.

    Bypasses external native C++/CUDA runtime dependencies (e.g. torch DLL issues on Windows)
    while providing fast, consistent 384-dimensional cosine vector similarity for ChromaDB.
    """

    def __init__(self, n_features: int = 384):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=True,
            norm="l2",
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.vectorizer.transform([text])
        return vector.toarray()[0].tolist()


def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    use_openai: bool = False,
    openai_api_key: Optional[str] = None,
) -> Embeddings:
    """Get the embedding model instance.

    Args:
        model_name: Name/path of the embedding model.
        use_openai: If True, uses OpenAIEmbeddings instead of local.
        openai_api_key: Optional API key for OpenAI.

    Returns:
        Embeddings instance.
    """
    if use_openai and openai_api_key:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model_name if "embedding" in model_name else "text-embedding-3-small",
            api_key=openai_api_key,
        )

    return FastLocalEmbeddings()


def create_vectorstore(
    documents: List[Document],
    embeddings: Embeddings,
    persist_directory: Optional[Path | str] = CHROMA_PERSIST_DIR,
    collection_name: str = "rag_collection",
) -> Chroma:
    """Create and persist a Chroma vector store from document chunks.

    Args:
        documents: List of Document chunks to index.
        embeddings: Embeddings model.
        persist_directory: Directory path to persist Chroma DB files.
        collection_name: Name of the Chroma collection.

    Returns:
        Chroma vectorstore instance.
    """
    persist_dir_str = str(persist_directory) if persist_directory else None
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir_str,
        collection_name=collection_name,
    )


def load_vectorstore(
    embeddings: Embeddings,
    persist_directory: Path | str = CHROMA_PERSIST_DIR,
    collection_name: str = "rag_collection",
) -> Chroma:
    """Load an existing Chroma vector store from disk.

    Args:
        embeddings: Embeddings model used when indexing.
        persist_directory: Directory where Chroma DB is persisted.
        collection_name: Name of the Chroma collection.

    Returns:
        Chroma vectorstore instance.
    """
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=collection_name,
    )
