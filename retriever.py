"""
retriever.py
------------
Retrieval strategy manager for RAG.
Supports Top-k, MMR, and Hybrid similarity.
"""


def get_retriever(vectorstore, method: str):
    """
    Get retriever from vectorstore with strategy.
    Args:
        vectorstore: LangChain vectorstore
        method (str): "topk" | "mmr" | "hybrid"
    Returns:
        Retriever object
    """
    if method == "topk":
        return vectorstore.as_retriever(search_kwargs={"k": 5})

    elif method == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.7}
        )

    elif method == "hybrid":
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.7, "k": 8},
        )

    else:
        raise ValueError(f"Unsupported retrieval strategy: {method}")
