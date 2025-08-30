# memory_manager.py
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage


class MemoryManager:
    def __init__(self, llm, method="windowed"):
        """
        Memory manager for conversational context.
        Supports:
        - windowed   → keeps buffer of messages
        - summarized → running summary with LLM
        - hybrid     → buffer + summary
        """
        self.method = method
        self.llm = llm
        self.chat_history = ChatMessageHistory()
        self.summary = "" if method in ["summarized", "hybrid"] else None

    def add_message(self, role, content):
        """Save a user/ai message into memory."""
        if role == "user":
            self.chat_history.add_message(HumanMessage(content=content))
        else:
            self.chat_history.add_message(AIMessage(content=content))

    def get_context(self):
        """Return memory context to inject into prompts."""
        if self.method == "windowed":
            return {"chat_history": self.chat_history.messages}

        elif self.method == "summarized":
            return {"chat_history": self.summary}

        elif self.method == "hybrid":
            return {
                "chat_history": self.chat_history.messages,
                "summary": self.summary,
            }

    def update_summary(self):
        """Update summary with LLM if summarized/hybrid memory is enabled."""
        if self.method in ["summarized", "hybrid"]:
            # Convert history to text
            conversation_text = "\n".join(
                [f"{msg.type}: {msg.content}" for msg in self.chat_history.messages]
            )
            # Generate summary
            summary_prompt = f"Summarize the following conversation:\n\n{conversation_text}\n\nSummary:"
            response = self.llm.invoke(summary_prompt)
            self.summary = response.content
