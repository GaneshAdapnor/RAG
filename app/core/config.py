from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Production RAG API"
    VERSION: str = "1.0.0"
    
    # Storage settings
    DATA_DIR: str = "./data"
    FAISS_INDEX_PATH: str = "./data/faiss_index.bin"
    METADATA_PATH: str = "./data/metadata.json"
    
    # Embedding settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    
    # Chunking settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Retrieval settings
    TOP_K_RETRIEVAL: int = 3
    
    # LLM Settings
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL_NAME: str = "gpt-4o-mini"
    
    # Upload limits
    MAX_UPLOAD_SIZE_MB: int = 50

    # API Rate Limiting
    RATE_LIMIT_CALLS: int = 10
    RATE_LIMIT_PERIOD_SECONDS: int = 60
    
    class Config:
        env_file = ".env"

settings = Settings()
