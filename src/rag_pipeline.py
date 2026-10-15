"""RAG Pipeline Module for Document Question Answering.

Handles:
- PDF document loading (single file or directory)
- Text chunking with configurable chunk size and overlap
- Embeddings creation (HuggingFace/local or OpenAI)
- Ingestion and persistence into local ChromaDB vector store
- Context retrieval returning formatted chunks with source page numbers
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class RetrievedChunk:
    """Represents a retrieved context chunk with metadata and citation info."""

    content: str
    source: str
    page: int
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "score": self.score,
            "metadata": self.metadata,
        }


def load_pdf_document(file_path: Union[str, Path]) -> List[Document]:
    """Load a single PDF document.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects (one per page).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    loader = PyPDFLoader(str(path))
    return loader.load()


def load_pdf_directory(directory_path: Union[str, Path], glob: str = "*.pdf") -> List[Document]:
    """Load all PDF documents from a directory.

    Args:
        directory_path: Directory containing PDF files.
        glob: Glob pattern for matching PDF files.

    Returns:
        List of Document objects across all matched PDFs.
    """
    path = Path(directory_path)
    if not path.is_dir():
        raise NotADirectoryError(f"Directory not found: {path}")

    loader = PyPDFDirectoryLoader(str(path), glob=glob)
    return loader.load()


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split documents into smaller chunks for vector indexing.

    Args:
        documents: List of Document objects to split.
        chunk_size: Maximum character length for each chunk.
        chunk_overlap: Overlap length between adjacent chunks.

    Returns:
        List of chunked Document objects.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size}).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def get_default_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    use_openai: bool = False,
    openai_api_key: Optional[str] = None,
) -> Embeddings:
    """Initialize an embedding model.

    Args:
        model_name: HuggingFace model or OpenAI embedding model name.
        use_openai: Whether to use OpenAIEmbeddings.
        openai_api_key: Optional OpenAI API key if using OpenAI.

    Returns:
        LangChain Embeddings instance.
    """
    if use_openai and openai_api_key:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model_name if "embedding" in model_name else "text-embedding-3-small",
            api_key=openai_api_key,
        )

    # Use fast, self-contained local embeddings by default (avoids host C++/CUDA runtime DLL issues)
    from src.vectorstore import FastLocalEmbeddings

    return FastLocalEmbeddings()


def save_to_chroma(
    documents: List[Document],
    embeddings: Embeddings,
    persist_directory: Optional[Union[str, Path]] = None,
    collection_name: str = "rag_documents",
) -> Chroma:
    """Store document chunks into a ChromaDB vector store.

    Args:
        documents: List of Document chunks to store.
        embeddings: Embedding model instance.
        persist_directory: Directory where ChromaDB files will be persisted (None for in-memory).
        collection_name: Name of the Chroma collection.

    Returns:
        Chroma vectorstore instance.
    """
    persist_dir = str(Path(persist_directory).resolve()) if persist_directory else None
    try:
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name=collection_name,
        )
    except Exception:
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=None,
            collection_name=collection_name,
        )


def retrieve_context_chunks(
    vectorstore: Chroma,
    query: str,
    top_k: int = 4,
    with_scores: bool = False,
) -> List[RetrievedChunk]:
    """Retrieve top-k relevant context chunks with source and page numbers.

    Args:
        vectorstore: Initialized Chroma vector store instance.
        query: User question or search query.
        top_k: Number of most relevant chunks to return.
        with_scores: If True, uses similarity search with relevance scores.

    Returns:
        List of RetrievedChunk instances containing text, source file, and 1-based page number.
    """
    results: List[RetrievedChunk] = []

    if with_scores:
        docs_and_scores = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
        for doc, score in docs_and_scores:
            page_num = doc.metadata.get("page", 0) + 1  # Convert 0-indexed to 1-indexed
            source = doc.metadata.get("source", "Unknown")
            results.append(
                RetrievedChunk(
                    content=doc.page_content,
                    source=source,
                    page=page_num,
                    score=float(score) if score is not None else None,
                    metadata=doc.metadata,
                )
            )
    else:
        docs = vectorstore.similarity_search(query, k=top_k)
        for doc in docs:
            page_num = doc.metadata.get("page", 0) + 1
            source = doc.metadata.get("source", "Unknown")
            results.append(
                RetrievedChunk(
                    content=doc.page_content,
                    source=source,
                    page=page_num,
                    score=None,
                    metadata=doc.metadata,
                )
            )

    return results


class DocumentRAGPipeline:
    """High-level class coordinating document ingestion and retrieval."""

    def __init__(
        self,
        persist_directory: Optional[Union[str, Path]] = None,
        collection_name: str = "rag_documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
    ):
        self.persist_directory = Path(persist_directory).resolve() if persist_directory else None
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = get_default_embeddings(
            model_name=embedding_model_name,
            use_openai=use_openai,
            openai_api_key=openai_api_key,
        )
        self.vectorstore: Optional[Chroma] = None

        # Load existing collection if directory exists
        if self.persist_directory and self.persist_directory.exists():
            try:
                self.vectorstore = Chroma(
                    persist_directory=str(self.persist_directory),
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name,
                )
            except Exception:
                self.vectorstore = None

    def ingest_pdf(self, file_path: Union[str, Path]) -> int:
        """Ingest a single PDF document into the Chroma vector database.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Number of chunks created and indexed.
        """
        documents = load_pdf_document(file_path)
        chunks = chunk_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        if self.vectorstore is None:
            self.vectorstore = save_to_chroma(
                documents=chunks,
                embeddings=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
            )
        else:
            self.vectorstore.add_documents(chunks)

        return len(chunks)

    def ingest_directory(self, directory_path: Union[str, Path], glob: str = "*.pdf") -> int:
        """Ingest all PDFs in a directory into the Chroma vector database.

        Args:
            directory_path: Path to the directory containing PDFs.
            glob: Matching pattern.

        Returns:
            Number of chunks created and indexed.
        """
        documents = load_pdf_directory(directory_path, glob=glob)
        chunks = chunk_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        if self.vectorstore is None:
            self.vectorstore = save_to_chroma(
                documents=chunks,
                embeddings=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
            )
        else:
            self.vectorstore.add_documents(chunks)

        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        with_scores: bool = False,
    ) -> List[RetrievedChunk]:
        """Retrieve relevant context chunks with source and page numbers.

        Args:
            query: Question or search query.
            top_k: Number of chunks to retrieve.
            with_scores: Whether to compute relevance scores.

        Returns:
            List of RetrievedChunk instances.
        """
        if self.vectorstore is None:
            raise ValueError(
                "Vector store is empty or not initialized. Call ingest_pdf() or ingest_directory() first."
            )

        return retrieve_context_chunks(
            vectorstore=self.vectorstore,
            query=query,
            top_k=top_k,
            with_scores=with_scores,
        )
