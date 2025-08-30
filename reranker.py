"""
reranker.py
-----------
Optional document reranker for RAG using Cohere.
"""

from langchain_cohere import CohereRerank


def apply_reranker(config, query: str, docs):
    """
    Apply reranking if enabled.
    Args:
        config (dict): Configuration dictionary
        query (str): User query
        docs (List[Document]): Retrieved docs
    Returns:
        Reranked docs
    """
    if config.get("reranker", False):
        reranker = CohereRerank(model="rerank-english-v3.0")
        return reranker.compress_documents(query=query, documents=docs)
    return docs
