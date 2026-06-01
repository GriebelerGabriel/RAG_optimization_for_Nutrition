"""Language models for text generation."""
from typing import TYPE_CHECKING

from src.core.interfaces import BaseLLM

if TYPE_CHECKING:
    from src.core.config import LLMConfig

from .litellm import LiteLLM

__all__ = ["LiteLLM", "create_llm"]

LLM_REGISTRY: dict[str, type[BaseLLM]] = {
    "openai": LiteLLM,
    "gpt": LiteLLM,
    "ollama": LiteLLM,
    "llama": LiteLLM,
    "mistral": LiteLLM,
    "litellm": LiteLLM,
}

DEFAULT_MODELS: dict[str, str] = {
    "openai": "openai/gpt-3.5-turbo",
    "gpt": "openai/gpt-3.5-turbo",
    "ollama": "ollama/llama3.1:8b",
    "llama": "ollama/llama3.1:8b",
    "mistral": "ollama/mistral:7b",
    "litellm": "openai/gpt-3.5-turbo",
}


def create_llm(config: "LLMConfig") -> BaseLLM:
    """Factory function to create LLM from config."""
    llm_class = LLM_REGISTRY.get(config.type)
    if llm_class is None:
        raise ValueError(f"Unknown LLM type: {config.type}. Available: {list(LLM_REGISTRY.keys())}")

    model_name = config.model_name
    if not model_name or model_name == "":
        model_name = DEFAULT_MODELS.get(config.type, "openai/gpt-3.5-turbo")

    # Auto-prefix model name for legacy types that don't include provider
    if config.type == "openai" and not model_name.startswith("openai/"):
        model_name = f"openai/{model_name}"
    elif config.type == "ollama" and not model_name.startswith("ollama/"):
        model_name = f"ollama/{model_name}"

    return llm_class(
        model_name=model_name,
        temperature=config.temperature,
        seed=config.seed,
        max_tokens=config.max_tokens,
        system_prompt=config.system_prompt,
        context_template=config.context_template,
    )
