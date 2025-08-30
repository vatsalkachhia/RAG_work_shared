"""
chunking.py
-----------
Provides multiple text chunking strategies for RAG.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from semantic_chunker import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


def get_chunker(method: str, text):
    """Factory to return selected chunker."""
    if method == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_text(text)
    elif method == "fixed":
        return [text[i:i+500] for i in range(0, len(text), 500)]
    elif method == "sliding":
        # Sliding window with overlap
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        return splitter.split_text(text)
    elif method == "semantic":
        embeddings = HuggingFaceEmbeddings(model_name = "sentence-transforms/all-MiniLM-L6-v2")
        chunker = SemanticChunker(embeddings = embeddings)
        return  chunker.chunk(text)
    else:
        raise ValueError(f"Unsupported chunking method: {method}")
