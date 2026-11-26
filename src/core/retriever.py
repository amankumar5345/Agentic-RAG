import streamlit as st
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings

from src.core.citations import CITATION_RULE, RetrievedChunk, format_chunks


@st.cache_resource
def load_retriever() -> ContextualCompressionRetriever:
    embeddings = OllamaEmbeddings(model="bge-m3:latest")
    db = Chroma(
        persist_directory="our_database/chroma_storage",
        embedding_function=embeddings
    )
    base_retriever = db.as_retriever(search_kwargs={"k": 25})
    compressor = FlashrankRerank(
        model="ms-marco-TinyBERT-L-2-v2",
        top_n=7
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )


def retrieve_chunks(user_input: str, retriever=None) -> list[RetrievedChunk]:
    """Retrieve and label chunks [C1]..[Cn] with document/section metadata."""
    retriever = retriever or load_retriever()
    documents = retriever.invoke(user_input)
    chunks = []
    for index, document in enumerate(documents, start=1):
        metadata = getattr(document, "metadata", {}) or {}
        source = str(metadata.get("source") or metadata.get("doc") or "")
        doc = source.replace("\\", "/").split("/")[-1]
        chunks.append(
            RetrievedChunk(
                id=f"C{index}",
                text=document.page_content,
                doc=doc,
                section=str(metadata.get("section", "")),
            )
        )
    return chunks


@tool
def load_prompt(user_input: str) -> str:
    """Retrieve official documents relevant to the user's technical question within 300 words strictly.
    Args:
        user_input: The user's complete technical question. Use the original
            question, not a shortened summary.

    Use this whenever the user asks a factual or technical question about lithography processes,
    cleanroom safety, equipment maintenance, or operating procedures. or related to any type of such machines"""
    chunks = retrieve_chunks(user_input)
    context_text = format_chunks(chunks)
    prompt_instructions = f"""You are an expert, highly accurate AI assistant for Photolithography and Cleanroom Operations. Obey the following rules with absolute precision:
    RAG retrieval has already been used once.
    Answer using the retrieved context within 300 words strictly.
    {CITATION_RULE}
    User question:
    {user_input}

    Retrieved context:
    {context_text}

    Answer the user's question using only the retrieved context.
    """
    return prompt_instructions
