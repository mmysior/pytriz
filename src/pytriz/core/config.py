from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    # ==========================================
    # LLM Configuration
    # ==========================================
    DEFAULT_PROVIDER: str = "openrouter"
    DEFAULT_MODEL: str = "qwen/qwen3.6-35b-a3b"

    # ==========================================
    # Semantic Search Configuration
    # ==========================================
    EMBEDDING_PROVIDER: Literal["huggingface", "openai", "ollama"] = "huggingface"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ==========================================
    # Services Configuration
    # ==========================================
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ==========================================
    # API Keys & Secrets
    # ==========================================
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    TOGETHER_API_KEY: str = ""
    MISTRAL_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""


config = Config()
