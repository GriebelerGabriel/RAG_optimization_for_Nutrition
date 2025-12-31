"""OpenAI language models."""
import os
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

from src.core.interfaces import BaseLLM

load_dotenv()


class OpenAILLM(BaseLLM):
    """LLM using OpenAI's chat completion API."""
    
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 1024,
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
        self._client: OpenAI | None = None
    
    @property
    def client(self) -> OpenAI:
        """Lazy load OpenAI client."""
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self._client = OpenAI(api_key=api_key)
        return self._client
    
    def generate(self, prompt: str) -> str:
        """Generate text using OpenAI chat completion.
        
        Args:
            prompt: The input prompt.
            
        Returns:
            Generated text.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        return response.choices[0].message.content or ""
