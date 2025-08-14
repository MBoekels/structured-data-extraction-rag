import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path('config') / '.env'
load_dotenv(dotenv_path=env_path, override=True)

class Settings:
    """Loads and stores all configuration settings for the application."""
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY") # Keep this as Deepseek API requires a key
    USE_OPENAI: bool = os.getenv("USE_OPENAI", "False").lower() == "true"
    OPENAI_EMBEDDING_MODEL_ID: str = os.getenv("OPENAI_EMBEDDING_MODEL_ID", "text-embedding-ada-002")
    OPENAI_LLM_MODEL_ID: str = os.getenv("OPENAI_LLM_MODEL_ID", "gpt-3.5-turbo")

    VLLM_EMBEDDING_API_BASE: str = os.getenv("VLLM_EMBEDDING_API_BASE", "http://localhost:8000/v1")
    VLLM_EMBEDDING_MODEL_ID: str = os.getenv("VLLM_EMBEDDING_MODEL_ID", "Qwen/Qwen3-Embedding-0.6B")

    DEEPSEEK_API_BASE: str = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    DEEPSEEK_LLM_MODEL_ID: str = os.getenv("DEEPSEEK_LLM_MODEL_ID", "deepseek-reasoner")

    EMBEDDING_MODEL_ID: str = OPENAI_EMBEDDING_MODEL_ID if USE_OPENAI else VLLM_EMBEDDING_MODEL_ID

    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))

    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///data/rag_app.db")

settings = Settings()
