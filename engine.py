"""
engine.py
---------
Core RAG Engine orchestration.
Handles: chunking → embeddings → vectorstore → retrieval → rerank → memory → LLM query.
"""

from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from memory_manager import MemoryManager
from chunking import get_chunker
from embeddings import get_embedding_model
from vectorstore import build_vectorstore
from retriever import get_retriever
from llm_loader import get_llm
from reranker import apply_reranker

# Load environment variables
from dotenv import load_dotenv
load_dotenv()   # Reads .env and sets values into environment


class RAGEngine:
    """
    RAGEngine
    ---------
    Modular, pluggable RAG pipeline.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = {"max_history_turns": 6, **config}
        self.vectorstore = None
        self.llm = get_llm(self.config["llm"])
        self.memory = MemoryManager(self.llm, method=self.config.get("memory", "windowed"))

    # ----------------- Document Handling -----------------
    def build_knowledge_base(self, text: str):
        """Chunk text, embed, and build vectorstore."""
        chunks = get_chunker(self.config["chunking"], text)
        # change: change the name to get_embeddings
        embedding_model = get_embedding_model(self.config["embedding"])
        self.vectorstore = build_vectorstore(self.config["vectordb"], chunks, embedding_model)

    # ----------------- Query Pipeline -----------------
    def query(self, question: str) -> Dict[str, Any]:
        """End-to-end RAG query."""
        if not self.vectorstore:
            raise RuntimeError("Vectorstore not initialized. Call build_knowledge_base().")

        # Save user input
        self.memory.add_message("user", question)

        # Retrieval
        retriever = get_retriever(self.vectorstore, self.config["retrieval"])
        docs = retriever.invoke(question)

        # Reranker
        docs = apply_reranker(self.config, question, docs)

        # Build context
        context = "\n\n".join([doc.page_content for doc in docs]).strip()
        mem_context = self.memory.get_context()

        # Guard against hallucination
        if not context or len(context.split()) < 5:
            ai_text = "I don't know based on the provided docs."
            self.memory.add_message("ai", ai_text)
            return {"answer": ai_text, "sources": []}

        # Prompt
        prompt = f"""
        You are a precise, extractive assistant.
        Answer ONLY using information from the Context.
        If not found, reply EXACTLY: "I don't know based on the provided docs."

        Memory: {mem_context}
        Context: {context}

        Question: {question}
        Answer:
        """

        response = self.llm.invoke(prompt)
        ai_text = response.content.strip()
        self.memory.add_message("ai", ai_text)

        return {"answer": ai_text, "sources": [doc.metadata for doc in docs]}
