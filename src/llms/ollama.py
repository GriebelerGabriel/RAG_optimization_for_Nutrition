"""Ollama local language models."""
from typing import Any

import httpx

from src.core.interfaces import BaseLLM


class OllamaLLM(BaseLLM):
    """LLM using Ollama for local model inference."""
    
    def __init__(
        self, 
        model_name: str = "llama3.1:8b",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 600.0,
        **kwargs: Any
    ):
        super().__init__(
            model_name=model_name, 
            temperature=temperature,
            **kwargs
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.timeout_s = timeout_s
    
    def generate(self, prompt: str) -> str:
        """Generate text using Ollama API.
        
        Args:
            prompt: The input prompt.
            
        Returns:
            Generated text.
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }
        
        timeout = httpx.Timeout(self.timeout_s)

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
        except httpx.ConnectError as e:
            raise RuntimeError(
                "Could not connect to Ollama. "
                f"Is the Ollama server running at {self.base_url}? "
                "Start it (e.g. `ollama serve`) and ensure the model is available "
                f"(e.g. `ollama pull {self.model_name}`)."
            ) from e
        except httpx.ReadTimeout as e:
            raise RuntimeError(
                "Ollama request timed out while waiting for the model to generate a response. "
                f"Current timeout is {self.timeout_s:.0f}s. "
                "Try reducing `llm.max_tokens`, using a smaller model (e.g. `llama3.2:3b`), "
                "or increase the timeout in the LLM configuration."
            ) from e
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = f" Response: {e.response.text}"
            except Exception:
                detail = ""
            raise RuntimeError(
                f"Ollama request failed (status {e.response.status_code}) for model '{self.model_name}'.{detail}"
            ) from e
        
        return result.get("response", "")
    
    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
