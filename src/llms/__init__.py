"""Language models for text generation."""
from typing import TYPE_CHECKING

from src.core.interfaces import BaseLLM

if TYPE_CHECKING:
    from src.core.config import LLMConfig

from .openai import OpenAILLM
from .ollama import OllamaLLM

__all__ = ["OpenAILLM", "OllamaLLM", "create_llm"]

LLM_REGISTRY: dict[str, type[BaseLLM]] = {
    "openai": OpenAILLM,
    "gpt": OpenAILLM,
    "ollama": OllamaLLM,
    "llama": OllamaLLM,
    "mistral": OllamaLLM,
}

DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "gpt": "gpt-4o-mini",
    "ollama": "llama3.1:8b",
    "llama": "llama3.1:8b",
    "mistral": "mistral:7b",
}


def create_llm(config: "LLMConfig") -> BaseLLM:
    """Factory function to create LLM from config."""
    llm_class = LLM_REGISTRY.get(config.type)
    if llm_class is None:
        raise ValueError(f"Unknown LLM type: {config.type}. Available: {list(LLM_REGISTRY.keys())}")
    
    model_name = config.model_name
    if not model_name or model_name == "":
        model_name = DEFAULT_MODELS.get(config.type, "gpt-4o-mini")
    
    return llm_class(
        model_name=model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        system_prompt=config.system_prompt,
        context_template=config.context_template,
    )
