from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chroma_persist_dir: str = "./chroma_db"
    ocr_lang: str = "tr"
    ocr_confidence_threshold: float = 0.7
    chunk_size: int = 512
    chunk_overlap: int = 64
    relevance_threshold: float = 0.3

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
