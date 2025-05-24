from pathlib import Path
import os

class Settings:
    # Model configurations
    LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight alternative to DistilBERT
    
    # Data processing
    README_DIR = Path(os.getenv("README_DIR", Path(__file__).parent / "knowledge_base"))
    CHUNK_SIZE = 300  # Reduced for better performance
    CHUNK_OVERLAP = 50
    
    # Vector search
    SEARCH_TOP_K = 3
    SIMILARITY_THRESHOLD = 0.3
    
    # Generation parameters
    MAX_TOKENS = 150
    TEMPERATURE = 0.7
    TOP_K = 40
    TOP_P = 0.9
    
    # Caching
    CACHE_DIR = Path("./cache")
    MAX_CACHE_SIZE = 100
    ENABLE_QUERY_CACHE = True
    
    # Server configuration
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 4000))
    RELOAD = os.getenv("ENVIRONMENT", "production") == "development"
    
    # Resource limits
    MAX_MEMORY_MB = 1024  # 1GB limit
    MAX_CONCURRENT_REQUESTS = 4
    
    # Performance optimizations
    TORCH_THREADS = 2
    OMP_NUM_THREADS = 2
    TOKENIZERS_PARALLELISM = False

# Apply performance settings
os.environ["OMP_NUM_THREADS"] = str(Settings.TORCH_THREADS)
os.environ["TOKENIZERS_PARALLELISM"] = str(Settings.TOKENIZERS_PARALLELISM).lower()

settings = Settings()

# Create cache directory
settings.CACHE_DIR.mkdir(exist_ok=True)
