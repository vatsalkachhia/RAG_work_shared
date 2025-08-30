from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticChunker:
    def __init__(self, embedding_model, chunk_size: int = 500, chunk_overlap: int = 50, similarity_threshold: float = 0.75):
        """
        Simple semantic chunker.
        - embedding_model: must have .embed_documents() method (e.g., HuggingFaceEmbeddings)
        - chunk_size: max characters per chunk
        - chunk_overlap: overlap for context continuity
        - similarity_threshold: merge chunks if semantically similar
        """
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str) -> List[str]:
        """Chunk text semantically, merging highly similar chunks."""
        # Step 1: Baseline split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_text(text)

        if not chunks:
            return []

        # Step 2: Embed chunks
        embeddings = self.embedding_model.embed_documents(chunks)
        embeddings = np.array(embeddings)

        # Step 3: Merge adjacent chunks if semantically similar
        merged_chunks = [chunks[0]]
        last_embedding = embeddings[0]

        for i in range(1, len(chunks)):
            sim = cosine_similarity([last_embedding], [embeddings[i]])[0][0]
            if sim > self.similarity_threshold:
                # Merge with previous
                merged_chunks[-1] += " " + chunks[i]
            else:
                # Start new chunk
                merged_chunks.append(chunks[i])
                last_embedding = embeddings[i]

        return merged_chunks
