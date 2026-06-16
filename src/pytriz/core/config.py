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
    DEFAULT_PROVIDER: str = "openai"
    DEFAULT_MODEL: str = "gpt-4.1"

    # ==========================================
    # Semantic Search Configuration
    # ==========================================
    EMBEDDING_PROVIDER: Literal["local", "openai"] = "local"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ==========================================
    # API Keys & Secrets
    # ==========================================
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    TOGETHER_API_KEY: str = ""
    MISTRAL_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""


config = Config()
