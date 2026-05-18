"""Configuration management for RAG experiments."""
import functools
import hashlib
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

load_dotenv()
load_dotenv(".env.local")


class ChunkerConfig(BaseModel):
    """Configuration for chunking strategy."""
    type: str = "fixed"
    chunk_size: int = 512
    overlap: int = 50
    
    
class EmbedderConfig(BaseModel):
    """Configuration for embedding model."""
    type: str = "e5"
    model_name: str = "intfloat/multilingual-e5-base"


class RetrieverConfig(BaseModel):
    """Configuration for retrieval strategy."""
    type: str = "naive"
    top_k: int = 5
    score_threshold: float = 0.3
    collection_name: str = "nutrition_docs"
    persist_directory: str = "./chroma_db"
    use_config_hash_collection: bool = True


class RerankerConfig(BaseModel):
    """Configuration for reranking model."""
    type: str = "none"
    model_name: str = "BAAI/bge-reranker-base"
    top_k: int = 5


class LLMConfig(BaseModel):
    """Configuration for language model."""
    type: str = "openai"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1024
    system_prompt: str | None = None
    context_template: str | None = None


class Config(BaseModel):
    """Main configuration for RAG experiment."""
    experiment_name: str = "baseline"
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    embedding: EmbedderConfig = Field(default_factory=EmbedderConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
    
    def get_collection_name(self) -> str:
        """Generate unique collection name based on embedding model."""
        # Sanitize model name for collection name
        model_safe = self.embedding.model_name.replace("/", "_").replace("-", "_")
        return f"{self.retriever.collection_name}_{model_safe}"
    
    @functools.cached_property
    def config_hash(self) -> str:
        """Hash of config components that affect the vector store.

        Cached after first computation since config is immutable.
        """
        hash_components = {
            "chunker_type": self.chunker.type,
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.overlap,
            "embedder_type": self.embedding.type,
            "embedder_model": self.embedding.model_name,
        }
        hash_str = str(sorted(hash_components.items()))
        return hashlib.md5(hash_str.encode()).hexdigest()[:12]

    def get_unique_collection_name(self) -> str:
        """Generate unique collection name based on full config hash.

        This prevents cross-run contamination when chunking or preprocessing differs.
        """
        base_name = self.retriever.collection_name
        return f"{base_name}_{self.config_hash}"


class EnvSettings(BaseSettings):
    """Environment variables for API keys etc."""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    huggingface_token: str = ""
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_config(path: str | Path) -> Config:
    """Load configuration from YAML file."""
    return Config.from_yaml(path)


def get_env_settings() -> EnvSettings:
    """Get environment settings."""
    return EnvSettings()
