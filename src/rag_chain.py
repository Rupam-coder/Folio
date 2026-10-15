from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from src.config import DEFAULT_LLM_MODEL, OPENAI_API_KEY


RAG_SYSTEM_PROMPT = """You are a helpful and precise assistant for Question-Answering tasks.
Use the following pieces of retrieved context to answer the question.
If you do not know the answer based on the context, say that you do not know. Do not hallucinate.
Keep your response concise, well-structured, and factual.

Context:
{context}
"""


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0) + 1 if "page" in doc.metadata else "N/A"
        formatted.append(f"[Document {i} | Source: {source} (Page {page})]:\n{doc.page_content}")
    return "\n\n".join(formatted)


def create_rag_chain(
    retriever,
    model_name: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.0,
    api_key: str = OPENAI_API_KEY,
):
    """Build an LCEL RAG chain that answers questions using retrieved context.

    Args:
        retriever: Vector store retriever instance.
        model_name: Model name for ChatOpenAI.
        temperature: Model sampling temperature.
        api_key: OpenAI API key.

    Returns:
        Callable function or LCEL runnable taking {"question": str} and returning
        dict with 'answer' and 'source_documents'.
    """
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key or None,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    answer_chain = (
        {
            "context": lambda x: format_docs(x["documents"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    def rag_pipeline(query: str) -> Dict[str, Any]:
        retrieved_docs = retriever.invoke(query)
        answer = answer_chain.invoke({"documents": retrieved_docs, "question": query})
        return {
            "answer": answer,
            "source_documents": retrieved_docs,
        }

    return rag_pipeline
