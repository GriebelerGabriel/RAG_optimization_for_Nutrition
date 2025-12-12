"""Ollama local language models."""
from typing import Any

import httpx

from src.core.interfaces import BaseLLM


class OllamaLLM(BaseLLM):
    """LLM using Ollama for local model inference."""
    
    def __init__(
        self, 
        model_name: str = "llama3.1:8b",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        base_url: str = "http://localhost:11434",
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
        
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
        
        return result.get("response", "")
    
    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
