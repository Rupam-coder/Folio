import os
import re
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
# Also resolve and load .env located in backend/ if running from repository root
backend_env_path = Path(__file__).resolve().parent / ".env"
if backend_env_path.exists():
    load_dotenv(dotenv_path=backend_env_path)

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Runtime check for GROQ_API_KEY
if not GROQ_API_KEY:
    warnings.warn(
        "GROQ_API_KEY is not set! Please define GROQ_API_KEY in backend/.env or your deployment environment variables.",
        UserWarning,
    )

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ---------------------------------------------------------------------------
# Embedding Initialization with Resilient Fallback
# ---------------------------------------------------------------------------
class FastLocalEmbeddings(Embeddings):
    """Normalized 384-dimensional hashing vectorizer for portable cosine search."""

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


def initialize_embeddings() -> Embeddings:
    """Initialize FastEmbed (ONNX) embeddings for lightweight, low-memory deployment (<100MB)."""
    try:
        import platform
        if platform.system() == "Windows":
            import subprocess, sys
            probe = subprocess.run([sys.executable, "-c", "import onnxruntime"], capture_output=True)
            if probe.returncode != 0:
                warnings.warn(
                    "Local Windows onnxruntime DLL issue detected. Falling back to FastLocalEmbeddings for local testing. "
                    "On Linux/Render, FastEmbed ONNX runs natively."
                )
                return FastLocalEmbeddings(n_features=384)

        return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        warnings.warn(f"FastEmbed load failed ({e}), falling back to FastLocalEmbeddings.")
        return FastLocalEmbeddings(n_features=384)


# Embedding model initialization
embeddings = initialize_embeddings()
embedding_model = embeddings

# Groq LLM client configured via LangChain ChatOpenAI interface
llm = ChatOpenAI(
    base_url=GROQ_BASE_URL,
    api_key=GROQ_API_KEY or "missing-groq-api-key",
    model=GROQ_MODEL,
    temperature=0.1,
)

# In-memory vectorstore session mapping: session_id -> Chroma instance
vectorstore_registry: Dict[str, Chroma] = {}
session_metadata: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# FastAPI Application & CORS Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Folio RAG API",
    description="Document QA RAG Backend powered by Groq Llama 3.1 & ChromaDB",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class UploadResponse(BaseModel):
    session_id: str
    filename: str
    chunk_count: int
    page_count: int
    message: str


class SourceItem(BaseModel):
    chunk_id: int
    source: str
    page: int
    content: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Active session ID returned by /upload")
    query: str = Field(..., description="User inquiry regarding the document")
    top_k: int = Field(default=4, ge=1, le=10, description="Top-k context chunks to retrieve")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceItem]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    """Health check and active session telemetry."""
    return {
        "status": "online",
        "service": "Folio RAG Backend",
        "model": GROQ_MODEL,
        "embedding": EMBEDDING_MODEL_NAME,
        "active_sessions": len(vectorstore_registry),
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Accept a PDF file, split text chunks (size=500, overlap=50), and index to in-memory Chroma."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported format. Only PDF files are supported.",
        )

    # Write uploaded PDF bytes to a temporary file for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Load PDF documents
        loader = PyPDFLoader(tmp_path)
        raw_docs = loader.load()

        if not raw_docs or not any(doc.page_content.strip() for doc in raw_docs):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract readable text from the provided PDF.",
            )

        # Standardize source metadata
        for doc in raw_docs:
            doc.metadata["source"] = file.filename

        # Split into semantic chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        chunked_docs = splitter.split_documents(raw_docs)

        # Create session ID and in-memory Chroma collection
        session_id = uuid.uuid4().hex
        collection_name = f"col_{session_id}"

        vectorstore = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            persist_directory=None,  # In-memory session store
            collection_name=collection_name,
        )

        vectorstore_registry[session_id] = vectorstore
        session_metadata[session_id] = {
            "filename": file.filename,
            "chunk_count": len(chunked_docs),
            "page_count": len(raw_docs),
        }

        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            chunk_count=len(chunked_docs),
            page_count=len(raw_docs),
            message="Document successfully processed and indexed into in-memory Chroma vectorstore.",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Retrieve top-4 chunks, ground prompt context, query Groq Llama 3.1, and return answer + citations."""
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GROQ_API_KEY is not configured. Please set GROQ_API_KEY in your backend/.env file or environment variables.",
        )

    session_id = request.session_id.strip()

    if session_id not in vectorstore_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session ID '{session_id}' not found. Please upload a PDF first.",
        )

    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    try:
        vectorstore = vectorstore_registry[session_id]

        # Retrieve top-k relevant document chunks
        retrieved_docs = vectorstore.similarity_search(query, k=request.top_k)

        source_items: List[SourceItem] = []
        context_blocks: List[str] = []

        for i, doc in enumerate(retrieved_docs, start=1):
            source_file = doc.metadata.get("source", "Uploaded Document")
            page_num = doc.metadata.get("page", 0) + 1  # 1-indexed

            # Normalize chunk whitespace so words flow as a readable paragraph
            normalized_content = re.sub(r"\s+", " ", doc.page_content).strip()

            source_items.append(
                SourceItem(
                    chunk_id=i,
                    source=source_file,
                    page=page_num,
                    content=normalized_content,
                )
            )

            context_blocks.append(
                f"[Source: {source_file} | Page: {page_num} | Chunk {i}]:\n{normalized_content}"
            )

        formatted_context = "\n\n".join(context_blocks)

        system_prompt = (
            "You are an expert Question-Answering research assistant for documents.\n"
            "Answer the user's question accurately using ONLY the provided document context below.\n"
            "For every key statement, cite the document name and page number (e.g. '[Source: document.pdf, Page 1]').\n"
            "If the context does not contain enough information to answer, state clearly that the document "
            "does not contain this information. Do not speculate or hallucinate.\n\n"
            f"Context:\n{formatted_context}"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]

        # Execute inference on Groq with graceful fallback
        try:
            response = await llm.ainvoke(messages)
            answer_text = response.content.strip()
        except Exception as llm_err:
            err_str = str(llm_err).lower()
            if "model_not_found" in err_str or "404" in err_str or "does not exist" in err_str:
                fallback_llm = ChatOpenAI(
                    base_url=GROQ_BASE_URL,
                    api_key=GROQ_API_KEY or "missing-groq-api-key",
                    model="openai/gpt-oss-20b",
                    temperature=0.1,
                )
                response = await fallback_llm.ainvoke(messages)
                answer_text = response.content.strip()
            else:
                raise llm_err

        return ChatResponse(
            session_id=session_id,
            answer=answer_text,
            sources=source_items,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
