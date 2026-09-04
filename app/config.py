import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
    VLLM_MODEL = os.getenv("VLLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/chroma_db")

    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", "8000"))
    
 

settings = Settings()

    
     
     