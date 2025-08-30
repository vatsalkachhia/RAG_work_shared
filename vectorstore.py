"""
vectorstore.py
--------------
Vector store builder for RAG.
Supports FAISS, Chroma, Pinecone.
"""

from langchain_community.vectorstores import FAISS, Chroma, Pinecone


def build_vectorstore(method: str, chunks, embedding_model):
    """
    Build vectorstore based on chosen backend.
    Args:
        method (str): "faiss" | "chroma" | "pinecone"
        chunks (List[str]): Preprocessed text chunks
        embeddings: Embedding model
    Returns:
        Vectorstore instance
    """
    if method == "faiss":
        return FAISS.from_texts(chunks, embedding_model)

    elif method == "chroma":
        return Chroma.from_texts(
            chunks, embedding_model, persist_directory="./chroma_store"
        )

    elif method == "pinecone":
        return Pinecone.from_texts(
            chunks, embedding_model, index_name="rag-prototype"
        )

    else:
        raise ValueError(f"Unsupported vector DB: {method}")
