import os
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from langchain_core.documents import Document

from src.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LLM_MODEL,
    OPENAI_API_KEY,
)
from src.document_loader import load_uploaded_pdf, split_documents
from src.rag_chain import create_rag_chain
from src.vectorstore import create_vectorstore, get_embedding_model

st.set_page_config(
    page_title="Document QA RAG Studio",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished card styling
st.markdown(
    """
    <style>
    .source-card {
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
        background-color: #f8f9fa;
        border-left: 4px solid #1E88E5;
    }
    .chunk-badge {
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "indexed_docs_meta" not in st.session_state:
    st.session_state.indexed_docs_meta = {}

# --- Top Banner ---
st.title("📄 Document QA RAG Studio")
st.markdown(
    "Upload PDF documents, customize text chunking, and query your content with citation-backed answers."
)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("1. 📂 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select PDF documents to parse and index into the vector database.",
    )

    st.header("2. ⚙️ Chunking Options")
    st.caption("Tune chunk dimensions to balance semantic completeness and retriever precision.")

    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=200,
        max_value=2500,
        value=DEFAULT_CHUNK_SIZE,
        step=50,
        help="Target length of each text chunk. Smaller chunks improve pinpoint retrieval; larger chunks preserve broader context.",
    )

    chunk_overlap = st.slider(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=500,
        value=DEFAULT_CHUNK_OVERLAP,
        step=25,
        help="Character overlap between consecutive chunks to maintain semantic continuity across boundaries.",
    )

    st.header("3. 🧠 Model & Retrieval")
    api_key_input = st.text_input(
        "OpenAI API Key (optional)",
        value=OPENAI_API_KEY,
        type="password",
        help="Required for LLM synthesis. Leave blank to run in direct chunk retrieval mode.",
    )

    model_name = st.selectbox(
        "OpenAI LLM Model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
    )

    top_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=1,
        max_value=8,
        value=4,
        help="Number of most similar text blocks to retrieve for each query.",
    )

    # Process and Index Button
    index_button = st.button("🚀 Process & Build Vector Index", use_container_width=True, type="primary")

    if index_button:
        if not uploaded_files:
            st.error("Please upload at least one PDF file first.")
        elif chunk_overlap >= chunk_size:
            st.error("Chunk Overlap must be strictly less than Chunk Size.")
        else:
            with st.spinner("Extracting PDF text and generating vector embeddings..."):
                try:
                    all_raw_docs: List[Document] = []
                    file_summaries = {}

                    for up_file in uploaded_files:
                        docs = load_uploaded_pdf(up_file)
                        all_raw_docs.extend(docs)
                        file_summaries[up_file.name] = len(docs)

                    # Split text with configured options
                    chunked_docs = split_documents(
                        all_raw_docs,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                    # Create embeddings & Chroma vectorstore
                    use_openai = bool(api_key_input.strip())
                    embeddings = get_embedding_model(
                        use_openai=use_openai,
                        openai_api_key=api_key_input.strip() if use_openai else None,
                    )
                    vectorstore = create_vectorstore(
                        documents=chunked_docs,
                        embeddings=embeddings,
                        persist_directory=None,  # In-memory session store for uploaded files
                    )

                    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

                    if api_key_input.strip():
                        # Build full LLM synthesis chain
                        rag_pipe = create_rag_chain(
                            retriever=retriever,
                            model_name=model_name,
                            api_key=api_key_input.strip(),
                        )
                    else:
                        # Fallback pipeline for retrieval-only inspection
                        def rag_pipe(query: str):
                            retrieved = retriever.invoke(query)
                            return {
                                "answer": (
                                    "*(OpenAI API Key not provided — showing top retrieved source blocks below. "
                                    "Provide an OpenAI API key in the sidebar for full synthesized answers.)*"
                                ),
                                "source_documents": retrieved,
                            }

                    st.session_state.rag_pipeline = rag_pipe
                    st.session_state.all_chunks = chunked_docs
                    st.session_state.indexed_docs_meta = file_summaries

                    st.success(
                        f"Indexed {len(chunked_docs)} chunks from {len(uploaded_files)} PDF(s) successfully!"
                    )
                except Exception as e:
                    st.error(f"Error during indexing: {str(e)}")

    st.divider()
    if st.session_state.all_chunks:
        st.caption(f"📊 **Active Vector Index**: {len(st.session_state.all_chunks)} chunks ready.")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# --- Main App Layout ---
tab_qa, tab_chunks = st.tabs(["💬 Query & Cited Answers", "🔍 Chunk Inspector"])

with tab_qa:
    # Render Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Render Retrieved Source Text Blocks
            sources = msg.get("sources", [])
            if sources:
                st.markdown("#### 📚 Retrieved Source Text Blocks")
                for i, doc in enumerate(sources, 1):
                    src_name = Path(doc.metadata.get("source", "Document")).name
                    page_num = doc.metadata.get("page", 0) + 1 if "page" in doc.metadata else "N/A"
                    char_count = len(doc.page_content)

                    with st.expander(f"Chunk #{i} | {src_name} (Page {page_num}) — {char_count} chars", expanded=False):
                        st.markdown(f"**Source:** `{src_name}` &nbsp;&nbsp;|&nbsp;&nbsp; **Page:** `{page_num}`")
                        st.text_area(
                            label=f"Content #{i}",
                            value=doc.page_content,
                            height=120,
                            disabled=True,
                            key=f"hist_{msg.get('id', '')}_{i}",
                        )

    # Chat Input
    query = st.chat_input("Type your question regarding the uploaded documents...")

    if query:
        # Check if pipeline exists
        if st.session_state.rag_pipeline is None:
            st.warning("⚠️ Please upload a PDF and click **Process & Build Vector Index** in the sidebar before querying.")
        else:
            # 1. User Message
            query_id = len(st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "user", "content": query, "id": f"q_{query_id}"})
            with st.chat_message("user"):
                st.markdown(query)

            # 2. Assistant Response
            with st.chat_message("assistant"):
                with st.spinner("Searching document index and synthesizing cited answer..."):
                    try:
                        res = st.session_state.rag_pipeline(query)
                        answer = res.get("answer", "No response generated.")
                        source_docs = res.get("source_documents", [])

                        st.markdown(answer)

                        if source_docs:
                            st.markdown("#### 📚 Retrieved Source Text Blocks")
                            for i, doc in enumerate(source_docs, 1):
                                src_name = Path(doc.metadata.get("source", "Document")).name
                                page_num = doc.metadata.get("page", 0) + 1 if "page" in doc.metadata else "N/A"
                                char_count = len(doc.page_content)

                                with st.expander(
                                    f"Chunk #{i} | {src_name} (Page {page_num}) — {char_count} chars",
                                    expanded=True,
                                ):
                                    st.markdown(f"**Source:** `{src_name}` &nbsp;&nbsp;|&nbsp;&nbsp; **Page:** `{page_num}`")
                                    st.text_area(
                                        label=f"Content #{i}",
                                        value=doc.page_content,
                                        height=120,
                                        disabled=True,
                                        key=f"curr_{query_id}_{i}",
                                    )

                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": source_docs,
                                "id": f"a_{query_id}",
                            }
                        )
                    except Exception as e:
                        err_text = f"❌ Retrieval/Generation Error: {str(e)}"
                        st.error(err_text)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": err_text, "id": f"err_{query_id}"}
                        )

with tab_chunks:
    st.subheader("📑 Chunking Preview & Index Explorer")
    if not st.session_state.all_chunks:
        st.info("Upload and index a PDF from the sidebar to inspect its generated chunks and chunk boundaries.")
    else:
        st.markdown(
            f"**Total Chunks Generated:** `{len(st.session_state.all_chunks)}` "
            f"| **Chunk Size:** `{chunk_size}` chars "
            f"| **Overlap:** `{chunk_overlap}` chars"
        )

        sample_idx = st.slider(
            "Select chunk index to inspect",
            min_value=1,
            max_value=len(st.session_state.all_chunks),
            value=1,
        )

        selected_chunk = st.session_state.all_chunks[sample_idx - 1]
        chunk_source = Path(selected_chunk.metadata.get("source", "Unknown")).name
        chunk_page = selected_chunk.metadata.get("page", 0) + 1 if "page" in selected_chunk.metadata else "N/A"

        col1, col2, col3 = st.columns(3)
        col1.metric("Chunk Index", f"#{sample_idx} of {len(st.session_state.all_chunks)}")
        col2.metric("Source File", chunk_source)
        col3.metric("Source Page", f"Page {chunk_page}")

        st.markdown("##### Chunk Text Content:")
        st.text_area(
            "Chunk Content",
            value=selected_chunk.page_content,
            height=200,
            disabled=True,
        )

        with st.expander("Show Metadata JSON"):
            st.json(selected_chunk.metadata)
