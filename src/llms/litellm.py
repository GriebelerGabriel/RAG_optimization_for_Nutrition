"""LiteLLM-based language model — unified interface to 100+ providers."""
import logging
from typing import Any

import litellm

from src.core.interfaces import BaseLLM

litellm.drop_params = True

logger = logging.getLogger(__name__)


class LiteLLM(BaseLLM):
    """LLM using litellm.completion() — supports OpenAI, Ollama, Anthropic, Google, Groq, etc.

    The model_name should be a litellm model string, e.g.:
      - "openai/gpt-4o-mini"
      - "openai/gpt-4o"
      - "ollama/llama3.1:8b"
      - "anthropic/claude-sonnet-4-20250514"
      - "gemini/gemini-2.0-flash"
      - "groq/llama-3.1-8b-instant"
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        num_retries: int = 3,
        timeout: float = 600.0,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_retries = num_retries
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                num_retries=self.num_retries,
                timeout=self.timeout,
            )
            return response.choices[0].message.content or ""
        except litellm.AuthenticationError as e:
            raise ValueError(
                f"Authentication failed for model '{self.model_name}'. "
                "Check your API key environment variables."
            ) from e
        except litellm.RateLimitError as e:
            raise RuntimeError(
                f"Rate limit exceeded for model '{self.model_name}'. "
                "Wait and retry, or switch to a different model/provider."
            ) from e
        except litellm.Timeout as e:
            raise RuntimeError(
                f"Request timed out for model '{self.model_name}'. "
                "Try reducing max_tokens or increase timeout settings."
            ) from e
        except litellm.APIConnectionError as e:
            provider = self.model_name.split("/")[0]
            raise RuntimeError(
                f"Could not connect to {provider} for model '{self.model_name}'. "
                "Ensure the service is running and accessible."
            ) from e
        except litellm.APIError as e:
            raise RuntimeError(
                f"LiteLLM API error for model '{self.model_name}': {e}"
            ) from e
