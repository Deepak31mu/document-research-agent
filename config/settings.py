from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from .constants import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    OPENAI_API_KEY: str

    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: List[str] = ALLOWED_TYPES

    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"

    VECTOR_SEARCH_K: int = 10
    HYBRID_RETRIEVER_WEIGHTS: List[float] = [0.4, 0.6]

    LOG_LEVEL: str = "INFO"

    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    GRADIO_SHARE: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        return Settings()
    except Exception as e:
        raise RuntimeError(
            "Failed to load settings. Ensure your .env file exists and OPENAI_API_KEY is set."
        ) from e


settings = get_settings()
