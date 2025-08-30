"""
embeddings.py
-------------
Embedding model loader for RAG.
Supports HuggingFace, OpenAI, and Instructor models.
"""

from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(method: str):
    """
    Factory function to load embeddings.
    Args:
        method (str): "huggingface" | "openai" | "instructor"
    Returns:
        LangChain-compatible embedding model
    """
    if method == "huggingface":
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    elif method == "openai":
        return OpenAIEmbeddings(model="text-embedding-3-small")

    elif method == "instructor":
        return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")

    else:
        raise ValueError(f"Unsupported embedding method: {method}")
