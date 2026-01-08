from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # IMPORTANT: field names must match env variables (case-insensitive)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_path: str = "./data/faiss_index"
    ollama_model: str = "phi3"
    ollama_url: str = "http://localhost:11434"

    # Load from .env and ignore unknown keys
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
