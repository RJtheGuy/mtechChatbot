from pathlib import Path
import os

class Settings:
    LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"


    
    README_DIR = Path(__file__).parent.parent / "knowledge_base"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Server
    HOST = "0.0.0.0"
    PORT = 8000
    RELOAD = True

settings = Settings()

