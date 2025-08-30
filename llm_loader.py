"""
llm_loader.py
-------------
LLM loader for RAG.
Supports OpenAI, Groq, HuggingFace pipelines.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFacePipeline


def get_llm(method: str):
    """
    Factory function to load LLM.
    Args:
        method (str): "openai" | "groq" | "huggingface"
    Returns:
        LLM instance
    """
    if method == "openai":
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    elif method == "groq":
        return ChatGroq(
            model="llama3-8b-8192",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    elif method == "huggingface":
        return HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-base", task="text2text-generation"
        )

    else:
        raise ValueError(f"Unsupported LLM: {method}")
