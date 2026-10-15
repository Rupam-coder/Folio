import tempfile
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def load_pdf(file_path: str | Path) -> List[Document]:
    """Load a PDF file and return extracted document pages.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects containing page content and metadata.
    """
    loader = PyPDFLoader(str(file_path))
    return loader.load()


def load_uploaded_pdf(uploaded_file) -> List[Document]:
    """Save an uploaded file from Streamlit to a temporary file and parse it.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        List of Document objects parsed from the uploaded PDF.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        documents = load_pdf(tmp_path)
        # Update metadata to reference original uploaded file name
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name
        return documents
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """Split documents into smaller chunks for vector indexing.

    Args:
        documents: List of Document objects to split.
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Overlap between adjacent chunks in characters.

    Returns:
        List of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_documents(documents)
