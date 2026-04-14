from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "Production RAG API"
    version: str = "1.0.0"
    api_prefix: str = ""

    data_dir: Path = Path("data")
    uploads_dir: Path = Path("data/uploads")
    index_dir: Path = Path("data/index")
    faiss_index_path: Path = Path("data/index/faiss.index")
    vector_metadata_path: Path = Path("data/index/vector_metadata.json")
    documents_path: Path = Path("data/index/documents.json")

    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 32

    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 100

    top_k_retrieval: int = 4
    max_top_k: int = 10
    search_min_score: float = 0.22

    openai_api_key: str | None = None
    llm_model_name: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 500
    enable_extractive_fallback: bool = True

    rate_limit_calls: int = 10
    rate_limit_period_seconds: int = 60

    max_upload_bytes: int = 20 * 1024 * 1024
    log_level: str = "INFO"

    supported_extensions: list[str] = Field(default_factory=lambda: [".pdf", ".txt"])

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
