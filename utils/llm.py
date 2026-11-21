"""LLM factory. Provider and model come from config.yaml, not code."""

from typing import Any, Dict

from langchain.chat_models import init_chat_model


def build_llm(llm_config: Dict[str, Any]):
    """Create a chat model (Groq by default; GROQ_API_KEY read from the env)."""
    return init_chat_model(
        llm_config["model_name"],
        model_provider=llm_config.get("model_provider", "groq"),
        temperature=llm_config.get("temperature", 0),
    )
